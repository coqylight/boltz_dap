#!/usr/bin/env python3
"""Monkey-patch Boltz2 forward to save tensor checkpoints, then run boltz predict.

Usage:
  python diag_original_checkpoints.py <yaml> <out_dir> [recycling_steps]
"""
import sys
import os
import torch

out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/diag_orig"
os.makedirs(out_dir, exist_ok=True)

def _ckp(label, z, s):
    zf = z.float()
    sf = s.float()
    print(f"    [CKP] [{label}]  z: mean={zf.mean():.6f} std={zf.std():.4f} "
          f"absmax={zf.abs().max():.4f}  shape={list(z.shape)}  |  "
          f"s: mean={sf.mean():.6f} std={sf.std():.4f} "
          f"absmax={sf.abs().max():.4f}  shape={list(s.shape)}")
    return {
        "z": z.cpu().to(torch.bfloat16),
        "s": s.cpu().to(torch.bfloat16),
    }


def make_patched_forward(original_forward, save_dir):
    def patched_forward(self, feats, recycling_steps=0, num_sampling_steps=None,
                        multiplicity_diffusion_train=1, diffusion_samples=1,
                        max_parallel_samples=None, run_confidence_sequentially=False):
        checkpoints = {}
        
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            s_inputs = self.input_embedder(feats)
            s_init = self.s_init(s_inputs)
            
            # Sub-checkpoint: s_inputs
            sf = s_inputs.float()
            print(f"    [CKP] [sub/s_inputs]  s_inputs: mean={sf.mean():.6f} std={sf.std():.4f} absmax={sf.abs().max():.4f}  shape={list(sf.shape)}")
            checkpoints["sub/s_inputs"] = {
                "s": s_inputs.cpu().to(torch.bfloat16),
                "z": torch.zeros(1),
            }
            
            # Sub-checkpoint: s_init
            sf = s_init.float()
            print(f"    [CKP] [sub/s_init]  s_init: mean={sf.mean():.6f} std={sf.std():.4f} absmax={sf.abs().max():.4f}  shape={list(sf.shape)}")
            checkpoints["sub/s_init"] = {
                "s": s_init.cpu().to(torch.bfloat16),
                "z": torch.zeros(1),
            }
            
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            # Sub-checkpoint: z_outer_product
            checkpoints["sub/z_outer_product"] = _ckp("sub/z_outer_product", z_init, s_init)
            
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            # Sub-checkpoint: z_after_rel_pos
            checkpoints["sub/z_after_rel_pos"] = _ckp("sub/z_after_rel_pos", z_init, s_init)
            
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())
            if self.bond_type_feature:
                z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
            # Sub-checkpoint: z_after_token_bonds
            checkpoints["sub/z_after_token_bonds"] = _ckp("sub/z_after_token_bonds", z_init, s_init)
            
            z_init = z_init + self.contact_conditioning(feats)
            # Sub-checkpoint: z_after_contact_cond
            checkpoints["sub/z_after_contact_cond"] = _ckp("sub/z_after_contact_cond", z_init, s_init)
            
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
            checkpoints["init"] = _ckp("init", z_init, s_init)
            
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]
            
            if self.run_trunk_and_structure:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        self.training
                        and self.structure_prediction_training
                        and (i == recycling_steps)
                    ):
                        if (self.training and (i == recycling_steps) 
                            and torch.is_autocast_enabled()):
                            torch.clear_autocast_cache()
                        
                        s = s_init + self.s_recycle(self.s_norm(s))
                        z = z_init + self.z_recycle(self.z_norm(z))
                        checkpoints[f"R{i}/after_recycle"] = _ckp(f"R{i}/after_recycle", z, s)
                        
                        if self.use_templates:
                            if self.is_template_compiled and not self.training:
                                template_module = self.template_module._orig_mod
                            else:
                                template_module = self.template_module
                            z = z + template_module(z, feats, pair_mask, use_kernels=self.use_kernels)
                            checkpoints[f"R{i}/after_template"] = _ckp(f"R{i}/after_template", z, s)
                        
                        if self.is_msa_compiled and not self.training:
                            msa_module = self.msa_module._orig_mod
                        else:
                            msa_module = self.msa_module
                        z = z + msa_module(z, s_inputs, feats, use_kernels=self.use_kernels)
                        checkpoints[f"R{i}/after_msa"] = _ckp(f"R{i}/after_msa", z, s)
                        
                        if self.is_pairformer_compiled and not self.training:
                            pairformer_module = self.pairformer_module._orig_mod
                        else:
                            pairformer_module = self.pairformer_module
                        s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask,
                                                use_kernels=self.use_kernels)
                        checkpoints[f"R{i}/after_pairformer"] = _ckp(f"R{i}/after_pairformer", z, s)
            
            # Save checkpoints
            ckpt_path = os.path.join(save_dir, "trunk_checkpoints.pt")
            torch.save(checkpoints, ckpt_path)
            ckpt_size = os.path.getsize(ckpt_path) / 1e6
            print(f"\n    [CKP] Saved {len(checkpoints)} checkpoints to {ckpt_path} ({ckpt_size:.0f} MB)")
            
            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram, "s": s, "z": z}
            
            if (self.run_trunk_and_structure
                and ((not self.training) or self.confidence_prediction)
                and (not self.skip_run_structure)):
                if self.checkpoint_diffusion_conditioning and self.training:
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        torch.utils.checkpoint.checkpoint(
                            self.diffusion_conditioning, s, z,
                            relative_position_encoding, feats,
                        )
                    )
                else:
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        self.diffusion_conditioning(
                            s_trunk=s, z_trunk=z,
                            relative_position_encoding=relative_position_encoding,
                            feats=feats,
                        )
                    )
                diffusion_conditioning = {
                    "q": q, "c": c, "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                }
                
                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module.sample(
                        s_trunk=s.float(), s_inputs=s_inputs.float(),
                        feats=feats, num_sampling_steps=num_sampling_steps,
                        atom_mask=feats["atom_pad_mask"].float(),
                        multiplicity=diffusion_samples,
                        max_parallel_samples=max_parallel_samples,
                        steering_args=self.steering_args,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)
                
                if self.predict_bfactor:
                    pbfactor = self.bfactor_module(s)
                    dict_out["pbfactor"] = pbfactor
            
            if self.training and self.confidence_prediction:
                assert len(feats["coords"].shape) == 4
                assert feats["coords"].shape[1] == 1
            
            if self.training and self.structure_prediction_training:
                atom_coords = feats["coords"]
                B, K, L = atom_coords.shape[0:3]
                atom_coords = atom_coords.reshape(B * K, L, 3)
                atom_coords = atom_coords.repeat_interleave(
                    multiplicity_diffusion_train // K, 0
                )
                feats["coords"] = atom_coords
                
                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module(
                        s_trunk=s.float(), s_inputs=s_inputs.float(),
                        feats=feats, multiplicity=multiplicity_diffusion_train,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)
            elif self.training:
                feats["coords"] = feats["coords"].squeeze(1)
        
        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(), s=s.detach(), z=z.detach(),
                    x_pred=(dict_out["sample_atom_coords"].detach()
                            if not self.skip_run_structure
                            else feats["coords"].repeat_interleave(diffusion_samples, 0)),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"][:, :, :, 0].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )
        
        if self.affinity_prediction:
            # Just call original for this complex part
            pass
        
        return dict_out
    
    return patched_forward


# Monkey-patch before boltz loads
from boltz.model.models.boltz2 import Boltz2
Boltz2.forward = make_patched_forward(Boltz2.forward, out_dir)
print(f"[DIAG] Patched Boltz2.forward to save checkpoints to {out_dir}/trunk_checkpoints.pt")

# Now run boltz predict via CLI
yaml_path = sys.argv[1]
recycling_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
diffusion_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1

sys.argv = [
    "boltz", "predict",
    yaml_path,
    "--out_dir", out_dir,
    "--recycling_steps", str(recycling_steps),
    "--diffusion_samples", str(diffusion_samples),
    "--override",
    "--num_workers", "2",
]

from boltz.main import cli
cli()
