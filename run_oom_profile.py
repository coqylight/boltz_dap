#!/usr/bin/env python3
"""
OOM profiler for original Boltz2.
Runs predict on a given input YAML with per-module memory hooks so we can see
exactly where memory peaks and where OOM happens (with full traceback).
"""
import sys, os, gc, time, traceback
import torch
import torch.cuda

# ── argument parsing ─────────────────────────────────────────────────────────
input_yaml = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/oom_profile_out"

print("=" * 80)
print("BOLTZ2 OOM MEMORY PROFILER")
print("=" * 80)
print(f"Input:  {input_yaml}")
print(f"Output: {out_dir}")
print(f"GPU:    {torch.cuda.get_device_name(0)}")
print(f"VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
print("=" * 80)

# ── Memory tracking infrastructure ──────────────────────────────────────────
_mem_log = []          # list of (timestamp, label, alloc_mb, peak_mb)
_start_time = None

def _log_mem(label):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    peak  = torch.cuda.max_memory_allocated(0) / 1024**2
    elapsed = time.time() - _start_time
    _mem_log.append((elapsed, label, alloc, peak))
    print(f"  [MEM] {elapsed:7.1f}s | alloc={alloc:8.0f}MB | peak={peak:8.0f}MB | {label}", flush=True)

def _reset_peak():
    torch.cuda.reset_peak_memory_stats(0)

# ── Hook machinery: register forward hooks on named modules ─────────────────
_hooks = []

def _make_hook(name, position="after"):
    """Create a forward hook (or pre-hook) that logs memory."""
    if position == "before":
        def hook_fn(module, input):
            _log_mem(f"BEFORE {name}")
        return hook_fn
    else:
        def hook_fn(module, input, output):
            _log_mem(f"AFTER  {name}")
        return hook_fn

def install_hooks(model):
    """Walk model and register memory-tracking hooks on key submodules."""
    targets = {}  # name -> module

    # Find key modules by their class names and path
    for name, mod in model.named_modules():
        cname = mod.__class__.__name__

        # Top-level components
        if name == "input_embedder":
            targets["input_embedder"] = mod
        elif name == "template_module":
            targets["template_module"] = mod
        elif name == "msa_module":
            targets["msa_module"] = mod
        elif name == "pairformer_module":
            targets["pairformer_module"] = mod
        elif name == "distogram_module":
            targets["distogram_module"] = mod
        elif name == "confidence_module":
            targets["confidence_module"] = mod
        elif name == "diffusion_module" or name == "score_network":
            targets[name] = mod
        elif name == "structure_module":
            targets["structure_module"] = mod
        elif name == "atom_encoder":
            targets["atom_encoder"] = mod
        elif name == "atom_decoder":
            targets["atom_decoder"] = mod
        elif name == "token_to_atom_decoder":
            targets["token_to_atom_decoder"] = mod

        # Pairformer layer-level: track first, mid, and last layers
        if "pairformer_module.layers." in name and name.endswith(".tri_mul_out"):
            layer_idx = name.split(".layers.")[1].split(".")[0]
            if layer_idx in ("0", "7", "15"):
                targets[f"pairformer.L{layer_idx}.tri_mul_out"] = mod
        if "pairformer_module.layers." in name and name.endswith(".tri_mul_in"):
            layer_idx = name.split(".layers.")[1].split(".")[0]
            if layer_idx in ("0", "7", "15"):
                targets[f"pairformer.L{layer_idx}.tri_mul_in"] = mod
        if "pairformer_module.layers." in name and name.endswith(".tri_att_start"):
            layer_idx = name.split(".layers.")[1].split(".")[0]
            if layer_idx in ("0", "7", "15"):
                targets[f"pairformer.L{layer_idx}.tri_att_start"] = mod
        if "pairformer_module.layers." in name and name.endswith(".tri_att_end"):
            layer_idx = name.split(".layers.")[1].split(".")[0]
            if layer_idx in ("0", "7", "15"):
                targets[f"pairformer.L{layer_idx}.tri_att_end"] = mod
        if "pairformer_module.layers." in name and name.endswith(".transition_z"):
            layer_idx = name.split(".layers.")[1].split(".")[0]
            if layer_idx in ("0", "7", "15"):
                targets[f"pairformer.L{layer_idx}.transition_z"] = mod

        # MSA module sub-ops
        if name == "msa_module.layers.0.opm":
            targets["msa.L0.opm"] = mod

        # Confidence pairformer
        if "confidence_module" in name and "pairformer" in name and name.endswith(".tri_mul_out"):
            targets[f"conf_pf.{name.split('.')[-2]}.tri_mul_out"] = mod

    print(f"\n  Installing memory hooks on {len(targets)} modules:")
    for tname in sorted(targets.keys()):
        print(f"    - {tname}")
    print()

    for tname, mod in targets.items():
        h1 = mod.register_forward_pre_hook(_make_hook(tname, "before"))
        h2 = mod.register_forward_hook(_make_hook(tname, "after"))
        _hooks.extend([h1, h2])

# ── Monkey-patch predict_step to get full OOM traceback ─────────────────────
def make_patched_predict_step(orig_fn):
    """Wrap predict_step to print full traceback + memory state on OOM."""
    import functools

    @functools.wraps(orig_fn)
    def patched(self, batch, batch_idx, dataloader_idx=0):
        _reset_peak()
        _log_mem("predict_step START")

        try:
            result = orig_fn(batch, batch_idx, dataloader_idx)
            _log_mem("predict_step END (success)")
            return result
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n" + "!" * 80, flush=True)
                print("  OOM CAUGHT — FULL TRACEBACK + MEMORY STATE", flush=True)
                print("!" * 80, flush=True)
                traceback.print_exc()

                alloc = torch.cuda.memory_allocated(0) / 1024**2
                reserved = torch.cuda.memory_reserved(0) / 1024**2
                peak = torch.cuda.max_memory_allocated(0) / 1024**2
                total = torch.cuda.get_device_properties(0).total_mem / 1024**2
                free = total - reserved
                print(f"\n  GPU memory at OOM:", flush=True)
                print(f"    Total:      {total:,.0f} MB ({total/1024:.1f} GB)", flush=True)
                print(f"    Allocated:  {alloc:,.0f} MB ({alloc/1024:.1f} GB)", flush=True)
                print(f"    Reserved:   {reserved:,.0f} MB ({reserved/1024:.1f} GB)", flush=True)
                print(f"    Peak alloc: {peak:,.0f} MB ({peak/1024:.1f} GB)", flush=True)
                print(f"    Free:       {free:,.0f} MB ({free/1024:.1f} GB)", flush=True)
                print(f"\n  Error: {e}", flush=True)
                print("!" * 80, flush=True)

                # Print memory timeline summary
                print("\n" + "=" * 80)
                print("  MEMORY TIMELINE SUMMARY (all hook events)")
                print("=" * 80)
                for ts, label, a, p in _mem_log:
                    print(f"  {ts:7.1f}s | alloc={a:8.0f}MB | peak={p:8.0f}MB | {label}")
                print("=" * 80 + "\n")

                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise

    return patched

# ── Main execution ──────────────────────────────────────────────────────────
def main():
    # Import boltz (uses the boltz venv)
    from boltz.main import BoltzRunner
    from boltz.model.models.boltz2 import Boltz2

    # Let Boltz2 set up everything via its normal CLI flow
    runner = BoltzRunner()
    runner.predict(
        data=input_yaml,
        out_dir=out_dir,
        use_msa_server=True,
        recycling_steps=3,
        sampling_steps=200,
        diffusion_samples=1,
    )

if __name__ == "__main__":
    main()
