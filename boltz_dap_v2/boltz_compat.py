"""Runtime compatibility patches for unmodified Boltz-2 installs.

Boltz-DAP CPU-offloads the diffusion token transformer bias to keep large
complexes from OOMing. Stock Boltz assumes that bias is already on the same
CUDA device as the attention activations, so we patch the narrow call sites
that consume the bias instead of requiring users to edit Boltz source files.
"""

from __future__ import annotations

from typing import Any

import torch


def _move_conditioning_tensors_to_device(
    diffusion_conditioning: dict[str, Any],
    device: torch.device,
) -> None:
    """Move small conditioning tensors to ``device`` in-place.

    ``token_trans_bias`` can be huge, so it stays CPU-resident and is moved one
    layer at a time by the DiffusionTransformer patch below.
    """

    for key, value in list(diffusion_conditioning.items()):
        if key == "token_trans_bias":
            continue
        if isinstance(value, torch.Tensor) and value.device != device:
            diffusion_conditioning[key] = value.to(device)


def _ensure_to_keys(
    module: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    diffusion_conditioning: dict[str, Any],
    device: torch.device,
) -> None:
    """Create ``to_keys`` locally when DAP broadcast omitted non-tensor values."""

    if callable(diffusion_conditioning.get("to_keys")):
        return

    from boltz.model.modules.encodersv2 import get_indexing_matrix, single_to_keys

    _, num_atoms, _ = feats["ref_pos"].shape
    window_queries = module.atoms_per_window_queries
    window_keys = module.atoms_per_window_keys
    num_windows = num_atoms // window_queries
    indexing_matrix = get_indexing_matrix(
        num_windows,
        window_queries,
        window_keys,
        device=device,
    )

    def to_keys(single: torch.Tensor) -> torch.Tensor:
        return single_to_keys(
            single,
            indexing_matrix=indexing_matrix,
            W=window_queries,
            H=window_keys,
        )

    diffusion_conditioning["to_keys"] = to_keys


def _patch_diffusion_module() -> bool:
    from boltz.model.modules.diffusionv2 import DiffusionModule

    if getattr(DiffusionModule.forward, "_boltz_dap_compat_patched", False):
        return False

    original_forward = DiffusionModule.forward

    def forward_with_dap_compat(
        self: torch.nn.Module,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        r_noisy: torch.Tensor,
        times: torch.Tensor,
        feats: dict[str, torch.Tensor],
        diffusion_conditioning: dict[str, Any],
        multiplicity: int = 1,
    ) -> torch.Tensor:
        device = r_noisy.device
        _move_conditioning_tensors_to_device(diffusion_conditioning, device)
        _ensure_to_keys(self, feats, diffusion_conditioning, device)
        return original_forward(
            self,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            r_noisy=r_noisy,
            times=times,
            feats=feats,
            diffusion_conditioning=diffusion_conditioning,
            multiplicity=multiplicity,
        )

    forward_with_dap_compat._boltz_dap_compat_patched = True
    forward_with_dap_compat._boltz_dap_original_forward = original_forward
    DiffusionModule.forward = forward_with_dap_compat
    return True


def _patch_diffusion_transformer() -> bool:
    from boltz.model.modules.transformersv2 import DiffusionTransformer

    if getattr(DiffusionTransformer.forward, "_boltz_dap_compat_patched", False):
        return False

    def forward_with_dap_compat(
        self: torch.nn.Module,
        a: torch.Tensor,
        s: torch.Tensor,
        bias: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        mask: torch.Tensor | None = None,
        to_keys: Any = None,
        multiplicity: int = 1,
    ) -> torch.Tensor:
        bias_is_list = isinstance(bias, (list, tuple))
        if self.pair_bias_attn and not bias_is_list:
            if bias is None:
                raise ValueError("DiffusionTransformer requires bias when pair_bias_attn=True")
            batch, rows, cols, channels = bias.shape
            num_layers = len(self.layers)
            bias = bias.view(batch, rows, cols, num_layers, channels // num_layers)

        for i, layer in enumerate(self.layers):
            if self.pair_bias_attn:
                if bias_is_list:
                    bias_l = bias[i]
                else:
                    bias_l = bias[:, :, :, i]

                if bias_l.device != a.device:
                    bias_l = bias_l.to(device=a.device, dtype=torch.float32)
                else:
                    bias_l = bias_l.float()
            else:
                bias_l = None

            if self.activation_checkpointing and self.training:
                a = torch.utils.checkpoint.checkpoint(
                    layer,
                    a,
                    s,
                    bias_l,
                    mask,
                    to_keys,
                    multiplicity,
                )
            else:
                a = layer(
                    a,
                    s,
                    bias_l,
                    mask,
                    to_keys,
                    multiplicity,
                )
        return a

    forward_with_dap_compat._boltz_dap_compat_patched = True
    DiffusionTransformer.forward = forward_with_dap_compat
    return True


def apply_template_t_chunk_size(
    model: torch.nn.Module, template_t_chunk_size: int | None
) -> None:
    """Set the template module's T-axis chunk size, if supported.

    Defined here (instead of imported from ``boltz.main``) so Boltz-DAP runs on an
    unmodified Boltz-2 install. On stock Boltz the template module has no
    ``template_t_chunk_size`` attribute, so this is a no-op; the chunking feature
    only takes effect on a Boltz build that supports it.
    """

    if template_t_chunk_size is None:
        return
    if not getattr(model, "use_templates", False):
        return
    tmpl = model.template_module
    if getattr(model, "is_template_compiled", False):
        tmpl = tmpl._orig_mod  # noqa: SLF001
    if hasattr(tmpl, "template_t_chunk_size"):
        tmpl.template_t_chunk_size = int(template_t_chunk_size)


# Backwards-compatible alias for the symbol previously imported from boltz.main.
_apply_template_t_chunk_size = apply_template_t_chunk_size


def apply_boltz_compat_patches() -> list[str]:
    """Apply Boltz-DAP compatibility patches and return patched component names."""

    patched = []
    if _patch_diffusion_module():
        patched.append("DiffusionModule.forward")
    if _patch_diffusion_transformer():
        patched.append("DiffusionTransformer.forward")
    return patched
