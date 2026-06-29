#!/usr/bin/env python3
"""Regression tests for root-only CPU gather and streamed distogram."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "boltz_dap_v2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _init_dist() -> tuple[int, int]:
    dist.init_process_group(backend="gloo", init_method="env://")
    import boltz_distributed.core as core

    core._DAP_INITIALIZED = True
    core._DAP_SIZE = dist.get_world_size()
    core._DAP_RANK = dist.get_rank()
    core._DAP_GROUP = None
    return core._DAP_RANK, core._DAP_SIZE


def _destroy_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def test_gather_to_rank0_cpu() -> None:
    from boltz_distributed.comm import gather, gather_to_rank0_cpu

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rows = 3
    original_size = world * local_rows - 1
    shard = (
        torch.arange(1 * local_rows * 5 * 2, dtype=torch.float32)
        .reshape(1, local_rows, 5, 2)
        .add(rank * 1000)
    )

    expected = gather(shard, dim=1, original_size=original_size).cpu()
    actual = gather_to_rank0_cpu(shard, dim=1, original_size=original_size, root=0)

    if rank == 0:
        assert actual is not None
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected)
    else:
        assert actual is None


class _TinyDistogram(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_distograms = 2
        self.num_bins = 3
        self.distogram = torch.nn.Linear(4, self.num_distograms * self.num_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + z.transpose(1, 2)
        return self.distogram(z).reshape(
            z.shape[0], z.shape[1], z.shape[2], self.num_distograms, self.num_bins
        )


def test_streamed_distogram_channel_matches_full_forward() -> None:
    from boltz_dap_v2.dap_trunk import _stream_distogram_channel_from_cpu

    torch.manual_seed(7)
    module = _TinyDistogram()
    z = torch.randn(1, 7, 7, 4)

    expected = module(z)[:, :, :, 0].detach()
    actual = _stream_distogram_channel_from_cpu(module, z.cpu(), channel=0, row_chunk=3)

    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected.cpu())


def test_distogram_contact_prob_stays_cpu_and_matches_full_softmax() -> None:
    from dap_confidence import _distogram_contact_prob_cpu

    torch.manual_seed(11)
    logits = torch.randn(2, 5, 5, 64)

    expected = torch.nn.functional.softmax(logits, dim=-1)[..., :20].sum(dim=-1)
    actual = _distogram_contact_prob_cpu(logits, row_chunk=2)

    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected)


def _tiny_ptm_feats() -> dict[str, torch.Tensor]:
    from boltz.data import const

    return {
        "asym_id": torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
        "atom_to_token": torch.eye(4, dtype=torch.float32).unsqueeze(0),
        "atom_pad_mask": torch.ones(1, 4, dtype=torch.float32),
        "token_pad_mask": torch.ones(1, 4, dtype=torch.float32),
        "frames_idx": torch.tensor(
            [[[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]], dtype=torch.long
        ),
        "mol_type": torch.full((1, 4), const.chain_type_ids["PROTEIN"], dtype=torch.long),
    }


def test_streamed_pae_helpers_match_full_confidence_utils() -> None:
    from boltz.model.layers.confidence_utils import (
        compute_aggregated_metric,
        compute_ptms,
    )
    from dap_confidence import (
        _compute_ptms_from_tm_expected,
        _pae_expected_from_logits,
        _pae_tm_expected_from_logits,
    )

    torch.manual_seed(19)
    logits = torch.randn(1, 4, 4, 64)
    x_pred = torch.randn(1, 4, 3)
    feats = _tiny_ptm_feats()

    expected_pae = compute_aggregated_metric(logits, end=32)
    actual_pae = _pae_expected_from_logits(logits)
    torch.testing.assert_close(actual_pae, expected_pae)

    tm_expected = _pae_tm_expected_from_logits(
        logits,
        feats["token_pad_mask"].sum(dim=-1, keepdim=True),
    )
    expected = compute_ptms(logits, x_pred, feats, multiplicity=1)
    actual = _compute_ptms_from_tm_expected(tm_expected, x_pred, feats, multiplicity=1)

    for actual_value, expected_value in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(actual_value, expected_value)
    for chain_id_1 in expected[4]:
        for chain_id_2 in expected[4][chain_id_1]:
            torch.testing.assert_close(
                actual[4][chain_id_1][chain_id_2],
                expected[4][chain_id_1][chain_id_2],
            )


def test_cpu_backed_diffusion_pairwise_and_token_bias_match_full_path() -> None:
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
    from boltz_dap_v2.dap_trunk import (
        _stream_pairwise_conditioning_to_cpu,
        _stream_token_trans_bias_from_cpu,
    )

    torch.manual_seed(13)
    dc = DiffusionConditioning(
        token_s=8,
        token_z=6,
        atom_s=8,
        atom_z=4,
        atoms_per_window_queries=2,
        atoms_per_window_keys=4,
        atom_encoder_depth=2,
        atom_encoder_heads=2,
        token_transformer_depth=3,
        token_transformer_heads=2,
        atom_decoder_depth=2,
        atom_decoder_heads=2,
        atom_feature_dim=3 + 1 + 128,
        conditioning_transition_layers=1,
        use_no_atom_char=True,
    ).eval()
    z = torch.randn(1, 5, 5, 6)
    rel = torch.randn(1, 5, 5, 6)

    expected_z_cond = dc.pairwise_conditioner(z, rel).detach()
    actual_z_cond = _stream_pairwise_conditioning_to_cpu(
        dc.pairwise_conditioner,
        z.cpu(),
        rel.cpu(),
        row_chunk=2,
        pin_memory=False,
    )
    assert actual_z_cond.device.type == "cpu"
    torch.testing.assert_close(actual_z_cond, expected_z_cond.cpu())

    expected_bias = torch.cat(
        [layer(expected_z_cond) for layer in dc.token_trans_proj_z], dim=-1
    ).detach()
    actual_bias = _stream_token_trans_bias_from_cpu(
        dc.token_trans_proj_z,
        actual_z_cond,
        row_chunk=2,
        pin_memory=False,
    )
    assert actual_bias.device.type == "cpu"
    torch.testing.assert_close(actual_bias, expected_bias.cpu())


def test_atom_encoder_accepts_cpu_backed_zcond() -> None:
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
    from boltz_dap_v2.dap_trunk import _run_atom_encoder_with_chunked_zcond

    torch.manual_seed(17)
    dc = DiffusionConditioning(
        token_s=8,
        token_z=6,
        atom_s=8,
        atom_z=4,
        atoms_per_window_queries=2,
        atoms_per_window_keys=4,
        atom_encoder_depth=2,
        atom_encoder_heads=2,
        token_transformer_depth=3,
        token_transformer_heads=2,
        atom_decoder_depth=2,
        atom_decoder_heads=2,
        atom_feature_dim=3 + 1 + 128,
        conditioning_transition_layers=1,
        use_no_atom_char=True,
    ).eval()
    feats = {
        "ref_pos": torch.randn(1, 4, 3),
        "atom_pad_mask": torch.ones(1, 4),
        "ref_space_uid": torch.arange(4).unsqueeze(0),
        "ref_charge": torch.randn(1, 4),
        "ref_element": torch.randn(1, 4, 128),
        "atom_to_token": torch.tensor(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]],
            dtype=torch.float32,
        ),
    }
    s = torch.randn(1, 3, 8)
    z_cond = torch.randn(1, 3, 3, 6)

    expected_q, expected_c, expected_p, _ = dc.atom_encoder(
        feats=feats,
        s_trunk=s,
        z=z_cond,
    )
    actual_q, actual_c, actual_p, _ = _run_atom_encoder_with_chunked_zcond(
        dc.atom_encoder,
        feats,
        s,
        z_cond.cpu(),
        chunk_rows=2,
    )

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_c, expected_c)
    torch.testing.assert_close(actual_p, expected_p)


def main() -> None:
    rank, _world = _init_dist()
    try:
        test_gather_to_rank0_cpu()
        if rank == 0:
            test_streamed_distogram_channel_matches_full_forward()
            test_distogram_contact_prob_stays_cpu_and_matches_full_softmax()
            test_streamed_pae_helpers_match_full_confidence_utils()
            test_cpu_backed_diffusion_pairwise_and_token_bias_match_full_path()
            test_atom_encoder_accepts_cpu_backed_zcond()
            print("rank0 CPU gather and streamed distogram tests passed")
    finally:
        _destroy_dist()


if __name__ == "__main__":
    required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [key for key in required if key not in os.environ]
    if missing:
        raise SystemExit(f"Run with torchrun; missing env vars: {', '.join(missing)}")
    main()
