#!/bin/bash
# ============================================================================
# OOM Memory Profiler for Original Boltz2
# Patches boltz2.py with _lg() memory tracing calls at every key forward-pass
# stage, and replaces the silent OOM catch with full traceback + memory dump.
# Runs pentamer then hexamer, restores original boltz2.py after.
#
# Submit: sbatch run_oom_profile.sbatch
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BOLTZ="/project/engvimmune/gleeai/boltz"
B2="$BOLTZ/src/boltz/model/models/boltz2.py"
OUT="/project/engvimmune/gleeai/boltz_output"

echo "=== OOM Memory Profiler ==="
echo "Backing up boltz2.py..."
cp "$B2" "$B2.bak_profile"

# ── Patch boltz2.py ──────────────────────────────────────────────────────────
uv run --project "$BOLTZ" python3 - "$B2" << 'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# === 1) Insert profiler helper functions ===
HELPERS = '''
# ===== OOM PROFILER =====
import time as _pt, traceback as _ptb
_pe, _pt0 = [], None
def _lg(label, dev=0):
    global _pt0
    import torch
    if _pt0 is None: _pt0 = _pt.time()
    a = torch.cuda.memory_allocated(dev)/1048576
    r = torch.cuda.memory_reserved(dev)/1048576
    p = torch.cuda.max_memory_allocated(dev)/1048576
    t = _pt.time()-_pt0
    _pe.append((t,label,a,r,p))
    print(f"  [MEM] {t:7.1f}s | alloc={a:8.0f}MB | rsrvd={r:8.0f}MB | peak={p:8.0f}MB | {label}",flush=True)
def _dump_oom(e):
    import torch
    print("\\n"+"!"*80,flush=True); print("  OOM TRACEBACK",flush=True); print("!"*80,flush=True)
    _ptb.print_exc()
    a=torch.cuda.memory_allocated(0)/1048576; r=torch.cuda.memory_reserved(0)/1048576
    p=torch.cuda.max_memory_allocated(0)/1048576; t=torch.cuda.get_device_properties(0).total_mem/1048576
    print(f"\\n  Total={t:.0f}MB  Alloc={a:.0f}MB  Reserved={r:.0f}MB  Peak={p:.0f}MB  Free={t-r:.0f}MB",flush=True)
    print(f"  Error: {e}",flush=True)
    print("\\n  MEMORY TIMELINE:"); mx=max(x[4] for x in _pe) if _pe else 0
    for ts,lb,ma,mr,mp in _pe:
        mark=" ◀◀ PEAK" if mp==mx else ""
        print(f"    {ts:7.1f}s | alloc={ma:8.0f}MB | peak={mp:8.0f}MB | {lb}{mark}")
    print("!"*80+"\\n",flush=True)
# ===== END PROFILER =====
'''

src = src.replace(
    'from boltz.model.models.boltz1 import Boltz1',
    'from boltz.model.models.boltz1 import Boltz1' + HELPERS,
    1
)

# === 2) Add _lg() calls at each stage in forward() ===
# Note: inside forward(), code is indented 12 spaces (            )
# Inside the recycling loop, code is indented 24 spaces (                        )

# After s_init (12-space indent)
src = src.replace(
    '            s_init = self.s_init(s_inputs)\n\n            # Initialize pairwise',
    '            s_init = self.s_init(s_inputs)\n            _lg("after s_init")\n\n            # Initialize pairwise',
    1
)

# After z_init (12-space indent)
src = src.replace(
    '            z_init = z_init + self.contact_conditioning(feats)\n\n            # Perform rounds',
    '            z_init = z_init + self.contact_conditioning(feats)\n            _lg(f"after z_init  shape={list(z_init.shape)}")\n\n            # Perform rounds',
    1
)

# Before/after template (28-space indent — inside 'if self.use_templates:')
src = src.replace(
    '                            z = z + template_module(\n                                z, feats, pair_mask, use_kernels=self.use_kernels\n                            )\n',
    '                            _lg(f"R{i} before template")\n                            z = z + template_module(\n                                z, feats, pair_mask, use_kernels=self.use_kernels\n                            )\n                            _lg(f"R{i} after template")\n',
    1
)

# Before/after msa (24-space indent)
src = src.replace(
    '                    z = z + msa_module(\n                        z, s_inputs, feats, use_kernels=self.use_kernels\n                    )\n',
    '                    _lg(f"R{i} before msa")\n                    z = z + msa_module(\n                        z, s_inputs, feats, use_kernels=self.use_kernels\n                    )\n                    _lg(f"R{i} after msa")\n',
    1
)

# Before/after pairformer (24-space indent)
src = src.replace(
    '                    s, z = pairformer_module(\n                        s,\n                        z,\n                        mask=mask,\n                        pair_mask=pair_mask,\n                        use_kernels=self.use_kernels,\n                    )\n',
    '                    _lg(f"R{i} before pairformer")\n                    s, z = pairformer_module(\n                        s,\n                        z,\n                        mask=mask,\n                        pair_mask=pair_mask,\n                        use_kernels=self.use_kernels,\n                    )\n                    _lg(f"R{i} after pairformer")\n',
    1
)

# Before/after distogram (12-space indent)
src = src.replace(
    '            pdistogram = self.distogram_module(z)\n',
    '            _lg("before distogram")\n            pdistogram = self.distogram_module(z)\n            _lg("after distogram — TRUNK DONE")\n',
    1
)

# Before diffusion_conditioning (non-checkpoint path, 20-space indent)
src = src.replace(
    '                else:\n                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (\n                        self.diffusion_conditioning(\n',
    '                else:\n                    _lg("before diffusion_conditioning")\n                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (\n                        self.diffusion_conditioning(\n',
    1
)

# After diffusion_conditioning (16-space indent)
src = src.replace(
    '                diffusion_conditioning = {\n',
    '                _lg("after diffusion_conditioning")\n                diffusion_conditioning = {\n',
    1
)

# Before structure_module.sample (16-space indent)
src = src.replace(
    '                with torch.autocast("cuda", enabled=False):\n                    struct_out = self.structure_module.sample(\n',
    '                _lg("before structure_module.sample (diffusion)")\n                with torch.autocast("cuda", enabled=False):\n                    struct_out = self.structure_module.sample(\n',
    1
)

# After struct_out (20-space indent -> 16-space for after)
src = src.replace(
    '                    dict_out.update(struct_out)\n\n                if self.predict_bfactor:',
    '                    dict_out.update(struct_out)\n                _lg("after structure_module.sample — DIFFUSION DONE")\n\n                if self.predict_bfactor:',
    1
)

# Before confidence_module (8-space indent)
src = src.replace(
    '        if self.confidence_prediction:\n            dict_out.update(\n                self.confidence_module(\n',
    '        if self.confidence_prediction:\n            _lg("before confidence_module")\n            dict_out.update(\n                self.confidence_module(\n',
    1
)

# After confidence_module
src = src.replace(
    '                )\n            )\n\n        if self.affinity_prediction:',
    '                )\n            )\n            _lg("after confidence_module")\n\n        if self.affinity_prediction:',
    1
)

# === 3) Replace silent OOM catch in predict_step ===
src = src.replace(
    '''        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e''',
    '''        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                _dump_oom(e)
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e''',
    1  # only first occurrence = predict_step
)

with open(path, 'w') as f:
    f.write(src)
print("  ✓ Patched boltz2.py with memory profiling")
PYEOF

# ── Verify ───────────────────────────────────────────────────────────────────
if grep -q "_lg\|_dump_oom" "$B2"; then
    echo "  ✓ Patch verified"
else
    echo "  ✗ Patch FAILED — restoring backup"
    cp "$B2.bak_profile" "$B2"
    exit 1
fi

# ── Run pentamer ─────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "RUN 1: PENTAMER (N=2595) — original Boltz2 + memory tracing"
echo "=============================================="

uv run --project "$BOLTZ" \
    boltz predict /project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml \
    --out_dir "$OUT/oom_profile_pentamer" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 || true

echo ""
echo "=============================================="
echo "PENTAMER COMPLETE"
echo "=============================================="

# ── Run hexamer ──────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "RUN 2: HEXAMER (N=3114) — original Boltz2 + memory tracing"
echo "=============================================="

uv run --project "$BOLTZ" \
    boltz predict /project/engvimmune/gleeai/boltz_input/1LP3_hexamer.yaml \
    --out_dir "$OUT/oom_profile_hexamer" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 || true

echo ""
echo "=============================================="
echo "HEXAMER COMPLETE"
echo "=============================================="

# ── Restore original ────────────────────────────────────────────────────────
cp "$B2.bak_profile" "$B2"
rm -f "$B2.bak_profile"
echo "  ✓ Restored original boltz2.py"

echo ""
echo "=============================================="
echo "ALL DONE — check [MEM] lines and OOM TRACEBACK above"
echo "=============================================="
