# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file is a public-facing version of the Aether orchestration
#  framework. Core logic and integrated modules are redacted for safety.
# ============================================================================

import os as _aos
import time as _atime


# ---------------------------
# 1) MPS watermark (safe gate)
# ---------------------------
def _aether_patch_mps_watermark():
    try:
        import torch

        if not hasattr(torch, "mps"):
            return
        ratio_s = _aos.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        if ratio_s is None:
            # Respect PyTorch default / user's runtime setting
            return
        try:
            r = float(ratio_s)
        except Exception:
            print(
                f"[MPS] Invalid PYTORCH_MPS_HIGH_WATERMARK_RATIO={ratio_s!r}; ignoring"
            )
            return
        if r <= 0.0:
            # "Unlimited" mode request: leave allocator untouched
            print("[MPS] watermark: unlimited requested (allocator untouched)")
            return
        try:



            torch.mps.set_per_process_memory_fraction(r)
            print(f"[MPS] watermark set to {r:.3f}")
        except Exception as e:
            print(f"[MPS] set_per_process_memory_fraction({r}) failed:", e)
    except Exception:
        # Non-fatal
        pass


if _aos.environ.get("AETHER_DISABLE_MPS_WM_PATCH", "0") != "1":
    _aether_patch_mps_watermark()


# -----------------------------
# 1a) MPS fast-path preferences
# -----------------------------
def _aether_mps_fastpath():
    try:
        import torch
    except Exception:
        return

    if not hasattr(torch, "mps"):
        return

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return
    try:
        if not backend.is_available():
            return
    except Exception:
        return

    if _aos.environ.get("AETHER_MPS_ASYNC", "1") != "0":
        setter = getattr(torch.mps, "set_graphs_sync_enabled", None)
        if setter is not None:
            try:
                setter(False)
                print("[MPS] async graph execution enabled")
            except Exception as e:
                print("[MPS] async graph execution toggle failed:", e)


if _aos.environ.get("AETHER_DISABLE_MPS_FASTPATH", "0") != "1":
    _aether_mps_fastpath()


# ------------------------------
# 1b) MPS warmup (pump & preheat)
# ------------------------------
def _aether_mps_warmup():
    auto_enable = _aos.environ.get("AETHER_MPS_WARMUP_AUTO", "1") != "0"
    steps_s = _aos.environ.get("AETHER_MPS_WARMUP_STEPS")
    if steps_s is None:
        if not auto_enable:
            return
        steps = 3
    else:
        try:
            steps = int(steps_s)
        except Exception:
            print(f"[MPS] Invalid AETHER_MPS_WARMUP_STEPS={steps_s!r}; ignoring")
            return

    if steps <= 0:
        return

    size_s = _aos.environ.get("AETHER_MPS_WARMUP_SIZE", "2048")
    try:
        size = int(size_s)
    except Exception:
        print(f"[MPS] Invalid AETHER_MPS_WARMUP_SIZE={size_s!r}; using 2048")
        size = 2048

    dtype_name = _aos.environ.get("AETHER_MPS_WARMUP_DTYPE", "float16").lower()

    try:
        import torch
    except Exception:
        return

    try:
        import torch.nn.functional as F
    except Exception:
        F = None

    if not hasattr(torch, "mps"):
        return

    backend = getattr(torch, "backends", None)
    if backend is not None and hasattr(torch.backends, "mps"):
        try:
            if not torch.backends.mps.is_available():
                return
        except Exception:
            return

    dtype_map = {
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": getattr(torch, "bfloat16", torch.float16),
    }
    dtype = dtype_map.get(dtype_name)
    if dtype is None:
        print(f"[MPS] Unknown dtype {dtype_name!r}; defaulting to float16")
        dtype = torch.float16

    sync = _aos.environ.get("AETHER_MPS_WARMUP_SYNC", "1") != "0"
    attn_enable = _aos.environ.get("AETHER_MPS_WARMUP_ATTENTION", "1") != "0"
    attn_steps_env = _aos.environ.get("AETHER_MPS_WARMUP_ATTENTION_STEPS")
    attn_heads = max(1, int(_aos.environ.get("AETHER_MPS_WARMUP_HEADS", "16")))
    attn_seq = max(1, int(_aos.environ.get("AETHER_MPS_WARMUP_SEQ", str(min(size, 1024)))))
    attn_dim = max(8, int(_aos.environ.get("AETHER_MPS_WARMUP_HEAD_DIM", "128")))

    def _round_up(v, multiple):
        return ((int(v) + multiple - 1) // multiple) * multiple

    align_enable = _aos.environ.get("AETHER_MPS_WARMUP_ALIGN", "1") != "0"
    if align_enable:
        size = max(size, _round_up(size, 8))
        attn_dim = max(attn_dim, _round_up(attn_dim, 16))
        attn_seq = max(attn_seq, _round_up(attn_seq, 16))

    try:
        attn_steps_default = max(1, min(steps, 4))
    except Exception:
        attn_steps_default = 1
    attn_steps = attn_steps_default
    if attn_steps_env:
        try:
            attn_steps = max(1, int(attn_steps_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_ATTENTION_STEPS={attn_steps_env!r}; using {attn_steps_default}"
            )
            attn_steps = attn_steps_default

    super_enable = _aos.environ.get("AETHER_MPS_WARMUP_SUPERCHARGE", "1") != "0"
    super_steps_env = _aos.environ.get("AETHER_MPS_WARMUP_SUPER_STEPS")
    super_batch_env = _aos.environ.get("AETHER_MPS_WARMUP_BATCH", "4")
    ff_dim_env = _aos.environ.get("AETHER_MPS_WARMUP_FF_DIM")
    ff_mult_env = _aos.environ.get("AETHER_MPS_WARMUP_FF_MULT", "4")
    ff_cap_env = _aos.environ.get("AETHER_MPS_WARMUP_FF_CAP")

    try:
        super_batch = max(1, int(super_batch_env))
    except Exception:
        print(
            f"[MPS] Invalid AETHER_MPS_WARMUP_BATCH={super_batch_env!r}; using 4"
        )
        super_batch = 4

    ff_mult_default = 4
    try:
        ff_mult = max(1, int(ff_mult_env))
    except Exception:
        print(
            f"[MPS] Invalid AETHER_MPS_WARMUP_FF_MULT={ff_mult_env!r}; using {ff_mult_default}"
        )
        ff_mult = ff_mult_default

    if ff_cap_env is not None:
        try:
            ff_cap = max(8, int(ff_cap_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_FF_CAP={ff_cap_env!r}; using max({size}*8, 8192)"
            )
            ff_cap = max(size * 8, 8192)
    else:
        ff_cap = max(size * 8, 8192)

    if ff_dim_env is not None:
        try:
            ff_dim = max(8, int(ff_dim_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_FF_DIM={ff_dim_env!r}; using derived"
            )
            ff_dim = max(8, min(size * ff_mult, ff_cap))
    else:
        ff_dim = max(8, min(size * ff_mult, ff_cap))
    if align_enable:
        ff_dim = max(ff_dim, _round_up(ff_dim, 32))

    super_steps_default = max(1, min(steps, 6))
    super_steps = super_steps_default
    if super_steps_env is not None:
        try:
            super_steps = max(1, int(super_steps_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_SUPER_STEPS={super_steps_env!r}; using {super_steps_default}"
            )
            super_steps = super_steps_default

    lora_enable = _aos.environ.get("AETHER_MPS_WARMUP_LORA", "1") != "0"
    lora_rank_env = _aos.environ.get("AETHER_MPS_WARMUP_LORA_RANK", "160")
    lora_alpha_env = _aos.environ.get("AETHER_MPS_WARMUP_LORA_ALPHA", "320")
    lora_batch_env = _aos.environ.get("AETHER_MPS_WARMUP_LORA_BATCH")
    lora_steps_env = _aos.environ.get("AETHER_MPS_WARMUP_LORA_STEPS")

    try:
        lora_rank = max(1, int(lora_rank_env))
    except Exception:
        print(
            f"[MPS] Invalid AETHER_MPS_WARMUP_LORA_RANK={lora_rank_env!r}; using 160"
        )
        lora_rank = 160

    try:
        lora_alpha = max(1.0, float(lora_alpha_env))
    except Exception:
        print(
            f"[MPS] Invalid AETHER_MPS_WARMUP_LORA_ALPHA={lora_alpha_env!r}; using 320"
        )
        lora_alpha = 320.0

    if lora_batch_env is None:
        lora_batch = super_batch
    else:
        try:
            lora_batch = max(1, int(lora_batch_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_LORA_BATCH={lora_batch_env!r}; using {super_batch}"
            )
            lora_batch = super_batch

    lora_steps_default = max(1, min(steps, 8))
    if lora_steps_env is None:
        lora_steps = lora_steps_default
    else:
        try:
            lora_steps = max(1, int(lora_steps_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_LORA_STEPS={lora_steps_env!r}; using {lora_steps_default}"
            )
            lora_steps = lora_steps_default

    bandwidth_enable = _aos.environ.get("AETHER_MPS_WARMUP_BANDWIDTH", "1") != "0"
    bandwidth_mb_env = _aos.environ.get("AETHER_MPS_WARMUP_BW_MB", "256")
    bandwidth_steps_env = _aos.environ.get("AETHER_MPS_WARMUP_BW_STEPS")

    try:
        bandwidth_mb = max(1, int(bandwidth_mb_env))
    except Exception:
        print(
            f"[MPS] Invalid AETHER_MPS_WARMUP_BW_MB={bandwidth_mb_env!r}; using 256"
        )
        bandwidth_mb = 256

    bandwidth_steps_default = max(1, steps)
    if bandwidth_steps_env is None:
        bandwidth_steps = bandwidth_steps_default
    else:
        try:
            bandwidth_steps = max(1, int(bandwidth_steps_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_BW_STEPS={bandwidth_steps_env!r}; using {bandwidth_steps_default}"
            )
            bandwidth_steps = bandwidth_steps_default

    try:
        device = torch.device("mps")
        with torch.no_grad():
            x = torch.randn((size, size), device=device, dtype=dtype)
            y = torch.randn((size, size), device=device, dtype=dtype)
            matmul_start = _atime.perf_counter()
            for _ in range(steps):
                z = torch.matmul(x, y)
                x, y = y, z
            if sync and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            matmul_elapsed = _atime.perf_counter() - matmul_start

            attn_elapsed = 0.0
            attn_tok_per_sec = 0.0
            if attn_enable:
                if F is None or not hasattr(F, "scaled_dot_product_attention"):
                    print("[MPS] attention warmup skipped: SDPA unavailable")
                else:
                    try:
                        q = torch.randn(
                            (1, attn_heads, attn_seq, attn_dim),
                            device=device,
                            dtype=dtype,
                        )
                        k = torch.randn_like(q)
                        v = torch.randn_like(q)
                        attn_start = _atime.perf_counter()
                        for _ in range(attn_steps):
                            F.scaled_dot_product_attention(
                                q, k, v, is_causal=True, dropout_p=0.0
                            )
                        if sync and hasattr(torch.mps, "synchronize"):
                            torch.mps.synchronize()
                        attn_elapsed = _atime.perf_counter() - attn_start
                        approx_tokens = float(attn_steps * attn_seq * attn_heads)
                        attn_tok_per_sec = approx_tokens / max(attn_elapsed, 1e-6)
                    except Exception as e:
                        print(f"[MPS] attention warmup skipped: {e}")

            ff_elapsed = 0.0
            ff_gflops_per_s = 0.0
            if super_enable:
                gelu_fn = None
                if F is not None and hasattr(F, "gelu"):
                    gelu_fn = F.gelu
                else:
                    gelu_cls = getattr(getattr(torch, "nn", None), "GELU", None)
                    if gelu_cls is not None:
                        gelu_inst = gelu_cls()

                        def gelu_fn(x):
                            return gelu_inst(x)

                if gelu_fn is None:

                    def gelu_fn(x):
                        return 0.5 * x * (
                            1.0
                            + torch.tanh(
                                0.7978845608 * (x + 0.044715 * x * x * x)
                            )
                        )

                try:
                    ff_input = torch.randn(
                        (super_batch, size), device=device, dtype=dtype
                    )
                    w1 = torch.randn((size, ff_dim), device=device, dtype=dtype)
                    w2 = torch.randn((ff_dim, size), device=device, dtype=dtype)
                    b1 = torch.randn((ff_dim,), device=device, dtype=dtype)
                    b2 = torch.randn((size,), device=device, dtype=dtype)
                    ff_start = _atime.perf_counter()
                    for _ in range(super_steps):
                        h = torch.matmul(ff_input, w1) + b1
                        h = gelu_fn(h)
                        ff_input = torch.matmul(h, w2) + b2
                    if sync and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    ff_elapsed = _atime.perf_counter() - ff_start
                    total_flops = float(super_steps) * 4.0 * super_batch * size * ff_dim
                    ff_gflops_per_s = total_flops / max(ff_elapsed, 1e-6) / 1e9
                except Exception as e:
                    print(f"[MPS] FFN supercharge skipped: {e}")

            lora_elapsed = 0.0
            lora_gflops_per_s = 0.0
            if lora_enable:
                try:
                    lora_input = torch.randn(
                        (lora_batch, size), device=device, dtype=dtype
                    )
                    lora_a = torch.randn((size, lora_rank), device=device, dtype=dtype)
                    lora_b = torch.randn((lora_rank, size), device=device, dtype=dtype)
                    lora_scale = lora_alpha / float(lora_rank)
                    lora_start = _atime.perf_counter()
                    for _ in range(lora_steps):
                        update = torch.matmul(lora_input, lora_a)
                        update = torch.matmul(update, lora_b) * lora_scale
                        lora_input = lora_input + update
                    if sync and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    lora_elapsed = _atime.perf_counter() - lora_start
                    total_flops = (
                        float(lora_steps) * 4.0 * lora_batch * size * lora_rank
                    )
                    lora_gflops_per_s = total_flops / max(lora_elapsed, 1e-6) / 1e9
                except Exception as e:
                    print(f"[MPS] LoRA boost skipped: {e}")

            bandwidth_elapsed = 0.0
            bandwidth_gbps = 0.0
            if bandwidth_enable:
                try:
                    elem_bytes = torch.tensor([], dtype=dtype).element_size()
                    target_bytes = bandwidth_mb * 1024 * 1024
                    scratch_cols = max(
                        size, int(target_bytes / max(elem_bytes, 1))
                    )
                    scratch = torch.randn(
                        (super_batch, scratch_cols), device=device, dtype=dtype
                    )
                    scratch_alt = torch.randn_like(scratch)
                    bandwidth_start = _atime.perf_counter()
                    for _ in range(bandwidth_steps):
                        torch.add(scratch, scratch_alt, alpha=1.0, out=scratch_alt)
                        torch.mul(scratch_alt, 1.0009765625, out=scratch)
                    if sync and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    bandwidth_elapsed = _atime.perf_counter() - bandwidth_start
                    bytes_touched = (
                        float(bandwidth_steps)
                        * scratch.numel()
                        * elem_bytes
                        * 3.0
                    )
                    bandwidth_gbps = bytes_touched / max(bandwidth_elapsed, 1e-6) / (1024 ** 3)
                except Exception as e:
                    print(f"[MPS] bandwidth sweep skipped: {e}")

        total_elapsed = (
            matmul_elapsed
            + attn_elapsed
            + ff_elapsed
            + lora_elapsed
            + bandwidth_elapsed
        )
        msg = (
            f"[MPS] warmup pumped {steps}x matmul (size={size}, dtype={dtype_name}) "
            f"in {matmul_elapsed:.3f}s"
        )
        if attn_enable and attn_elapsed > 0.0:
            msg += (
                f"; attention warmup {attn_steps}x (seq={attn_seq}, heads={attn_heads}, "
                f"dim={attn_dim}) in {attn_elapsed:.3f}s (~{attn_tok_per_sec:,.0f} tok/s)"
            )
        if super_enable and ff_elapsed > 0.0:
            msg += (
                f"; ff-supercharge {super_steps}x (batch={super_batch}, hidden={ff_dim}) "
                f"in {ff_elapsed:.3f}s (~{ff_gflops_per_s:,.1f} GFLOP/s)"
            )
        if lora_enable and lora_elapsed > 0.0:
            msg += (
                f"; lora-boost {lora_steps}x (batch={lora_batch}, rank={lora_rank}, alpha={lora_alpha:g}) "
                f"in {lora_elapsed:.3f}s (~{lora_gflops_per_s:,.1f} GFLOP/s)"
            )
        if bandwidth_enable and bandwidth_elapsed > 0.0:
            msg += (
                f"; bandwidth-sweep {bandwidth_steps}x (~{bandwidth_mb} MiB window) "
                f"in {bandwidth_elapsed:.3f}s (~{bandwidth_gbps:,.1f} GiB/s)"
            )
        msg += f" [total {total_elapsed:.3f}s]"
        print(msg)
    except Exception as e:
        print(f"[MPS] warmup failed: {e}")


if _aos.environ.get("AETHER_DISABLE_MPS_WARMUP", "0") != "1":
    _aether_mps_warmup()


# ---------------------------------------
# 2) Backward guard (retain_graph retry)
# ---------------------------------------
def _aether_backward_guard():
    try:
        import torch
    except Exception:
        return

    _orig_autograd_backward = torch.autograd.backward
    _orig_tensor_backward = torch.Tensor.backward

    def _retry_with_retain_autograd(
        tensors, grad_tensors=None, retain_graph=None, create_graph=False, inputs=None
    ):
        try:
            return _orig_autograd_backward(
                tensors,
                grad_tensors=grad_tensors,
                retain_graph=retain_graph,
                create_graph=create_graph,
                inputs=inputs,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            # Typical messages: "Trying to backward through the graph a second time"
            if ("second backward" in msg or "retain_graph" in msg) and not retain_graph:
                return _orig_autograd_backward(
                    tensors,
                    grad_tensors=grad_tensors,
                    retain_graph=True,
                    create_graph=create_graph,
                    inputs=inputs,
                )
            raise

    def _retry_with_retain_tensor(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ):
        try:
            return _orig_tensor_backward(
                self,
                gradient=gradient,
                retain_graph=retain_graph,
                create_graph=create_graph,
                inputs=inputs,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if ("second backward" in msg or "retain_graph" in msg) and not retain_graph:
                return _orig_tensor_backward(
                    self,
                    gradient=gradient,
                    retain_graph=True,
                    create_graph=create_graph,
                    inputs=inputs,
                )
            raise

    torch.autograd.backward = _retry_with_retain_autograd
    torch.Tensor.backward = _retry_with_retain_tensor
    print("[GUARD] backward guard enabled")


if _aos.environ.get("AETHER_BACKWARD_GUARD_DISABLED", "0") != "1":
    _aether_backward_guard()

# ---------------------------------------------------------
# 3) Light Paged-Attention wrapper for inference-time SDPA
# ---------------------------------------------------------
_AETHER_SDPA_ORIG = None
_AETHER_SDPA_WINDOW = int(_aos.environ.get("AETHER_SDPA_WINDOW", "0"))


def enable_tiled_sdpa(
    tiled_q: int = 0,
    tiled_k: int = 0,
    compute_in_fp32: bool = True,
    window_size: int = None,
):
    """
    Enable a light wrapper around torch.nn.functional.scaled_dot_product_attention.

    If window_size is provided (or env AETHER_SDPA_WINDOW>0), and Q len == 1 (typical during generation),
    we restrict K/V to the last `window_size` tokens (paged/streamed attention).

    tiled_q/tiled_k are accepted for API compatibility (no-ops here).
    """
    global _AETHER_SDPA_ORIG, _AETHER_SDPA_WINDOW
    try:
        import torch.nn.functional as F
    except Exception:
        return

    if window_size is not None:
        _AETHER_SDPA_WINDOW = int(window_size)

    if _AETHER_SDPA_ORIG is None:
        _AETHER_SDPA_ORIG = F.scaled_dot_product_attention

    def _sdpa_wrapper(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        w = _AETHER_SDPA_WINDOW
        # If generating (q_len==1) and window enabled, crop tail window
        if w and q.size(-2) == 1 and k.size(-2) > w:
            k = k[..., -w:, :]
            v = v[..., -w:, :]
            if attn_mask is not None and getattr(attn_mask, "dim", lambda: 0)() >= 2:
                attn_mask = attn_mask[..., -w:]
        if compute_in_fp32:
            qf, kf, vf = q.float(), k.float(), v.float()
            out = _AETHER_SDPA_ORIG(
                qf,
                kf,
                vf,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
            return out.to(q.dtype)
        else:
            return _AETHER_SDPA_ORIG(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

    F.scaled_dot_product_attention = _sdpa_wrapper
    print(
        f"[SDPA] wrapper enabled (window={_AETHER_SDPA_WINDOW}, fp32={compute_in_fp32})"
    )


def disable_tiled_sdpa():
    """Restore the original SDPA implementation."""
    global _AETHER_SDPA_ORIG
    try:
        import torch.nn.functional as F

        if _AETHER_SDPA_ORIG is not None:
            F.scaled_dot_product_attention = _AETHER_SDPA_ORIG
            print("[SDPA] wrapper disabled")
    finally:
        _AETHER_SDPA_ORIG = None


# === END AETHER_PATCH_PAGED_ATTEN_AND_GUARDS ===
# =============================================================================
# SpiralReality Proprietary – LicenseRef-SpiralReality-Proprietary
# SPDX-License-Identifier: LicenseRef-SpiralReality-Proprietary
# © 2025 SpiralReality（Ryō ∴ SpiralArchitect + collaborators）All rights reserved.
#
# Aether v2.8 – (MPS-only; M4 optimized, stable core)
#   • SDPA: Tiled + Sliding Window + Global tokens (online softmax, MPS-safe)
#   • GQA/MQA: kv_heads for K/V head sharing
#   • INT8-base + LoRA (custom) / PEFT-LoRA / Hybrid
#   • CPU-AdamW(8bit-ish) optional
#   • LVI (light): teacher-based Z-bias + low-frequency cache
#   • Intention Contrastive Loss (teacher-positive/negative)
#   • ReLoRA cycle (periodic merge→re-apply)
#   • Streaming dataset / Curriculum / TorchScript trace
# =============================================================================

import os
import time
import math
import json
import glob
import random
import types
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

import torch
from torch.amp import GradScaler


# === Aether injected: GaLore-like low-rank optimizer (matrix params only) ==============
class GaLoreAdamW(torch.optim.Optimizer):
    # AdamW in a projected (low-rank) space for 2D weight matrices (e.g., Linear).
    # For a weight W in R^{O×I}, keep two fixed orthonormal projectors:
    #   P_out in R^{O×r},  P_in in R^{I×r}
    # Maintain Adam states only for the r×r core (m_core, v_core).
    # Update rule (decoupled weight decay):
    #   g_core = P_out^T (grad W) P_in
    #   m_core <- beta1 m_core + (1-beta1) g_core
    #   v_core <- beta2 v_core + (1-beta2) g_core ⊙ g_core
    #   ΔW = P_out ( m_hat / (sqrt(v_hat)+eps) ) P_in^T
    #   W <- (1 - lr*wd)*W - lr*ΔW
    # Non-matrix params (bias/LayerNorm) fall back to standard AdamW states.
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        rank=64,
        seed=1337,
        device=None,
        dtype=None,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid lr")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rank = int(rank)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._galore = {}
        self._rng = torch.Generator(device="cpu").manual_seed(self.seed)

    @torch.no_grad()
    def _init_param(self, p):
        pid = id(p)
        if pid in self._galore:
            return
        dev = p.device if self.device is None else torch.device(self.device)
        dt = p.dtype if self.dtype is None else self.dtype
        st = {"t": 0, "matrix": False}
        if p.ndim == 2 and p.numel() >= self.rank * self.rank:
            O, I = p.shape
            r = min(self.rank, O, I)
            A = torch.randn((O, r), generator=self._rng, device=dev, dtype=dt)
            B = torch.randn((I, r), generator=self._rng, device=dev, dtype=dt)
            Q_out, _ = torch.linalg.qr(A, mode="reduced")
            Q_in, _ = torch.linalg.qr(B, mode="reduced")
            st["P_out"] = Q_out
            st["P_in"] = Q_in
            st["m"] = torch.zeros((r, r), device=dev, dtype=dt)
            st["v"] = torch.zeros((r, r), device=dev, dtype=dt)
            st["matrix"] = True
        else:
            st["m"] = torch.zeros_like(p, device=dev, dtype=dt)
            st["v"] = torch.zeros_like(p, device=dev, dtype=dt)
        self._galore[pid] = st

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._init_param(p)
                st = self._galore[id(p)]
                st["t"] += 1

                if wd != 0 and p.ndim >= 1:
                    p.mul_(1 - lr * wd)

                g = p.grad
                if st["matrix"]:
                    P_out = st["P_out"]
                    P_in = st["P_in"]
                    g_core = P_out.T @ g @ P_in
                    st["m"].mul_(beta1).add_(g_core, alpha=(1 - beta1))
                    st["v"].mul_(beta2).addcmul_(g_core, g_core, value=(1 - beta2))
                    m_hat = st["m"] / (1 - beta1 ** st["t"])
                    v_hat = st["v"] / (1 - beta2 ** st["t"])
                    core = m_hat / (v_hat.sqrt() + eps)
                    p.addmm_(P_out, core @ P_in.T, alpha=-lr)
                else:
                    st["m"].mul_(beta1).add_(g, alpha=(1 - beta1))
                    st["v"].mul_(beta2).addcmul_(g, g, value=(1 - beta2))
                    m_hat = st["m"] / (1 - beta1 ** st["t"])
                    v_hat = st["v"] / (1 - beta2 ** st["t"])
                    p.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-lr)

        return loss


# === end GaLore-like optimizer ===============================================


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

# === ANI-AI (numpy micro-agent; optional, opt-in via env) =====================
try:
    import numpy as _np

    _ANI_AI_AVAILABLE = True
except Exception:
    _ANI_AI_AVAILABLE = False


class _AetherNumpyAIController:
    """
    Lightweight bandit-like controller that reacts to numeric hazards and stabilizes training.
    - No structure changes: installed from TrainerBase.__init__ (opt-in via env).
    - Uses numpy only. Keeps small moving windows and picks actions via UCB.
    - Actions (all guarded & reversible): loss-scale backoff, dynamic grad-clip, LR backoff,
      skip-step, safe softmax/log_softmax patch, optimizer state sanitization.
    """

    def __init__(self, trainer):
        self.tr = trainer
        self.enabled = bool(int(os.environ.get("AETHER_AI_ENABLE", "0")))
        self.ucb_c = float(os.environ.get("AETHER_AI_UCB_C", "1.25"))
        self.min_ls = float(
            os.environ.get(
                "AETHER_AI_MIN_LOSS_SCALE",
                os.environ.get("AETHER_ANI_MIN_LOSS_SCALE", "0.015625"),
            )
        )
        self.ls_backoff = float(
            os.environ.get(
                "AETHER_AI_LS_BACKOFF",
                os.environ.get("AETHER_ANI_SCALE_BACKOFF", "0.5"),
            )
        )
        self.grad_clip_max = float(os.environ.get("AETHER_AI_MAX_CLIP", "1.0"))
        self.allow_lr_backoff = bool(
            int(os.environ.get("AETHER_AI_LR_BACKOFF_ENABLE", "1"))
        )
        self.lr_backoff = float(os.environ.get("AETHER_AI_LR_BACKOFF", "0.5"))
        self.safe_softmax_default = bool(
            int(os.environ.get("AETHER_AI_SAFE_SOFTMAX", "1"))
        )
        self.sanitize_opt_default = bool(
            int(os.environ.get("AETHER_AI_SANITIZE_OPT", "1"))
        )
        self.hazard_patience = int(os.environ.get("AETHER_AI_PATIENCE", "1"))
        self.cooldown = int(
            os.environ.get(
                "AETHER_AI_COOLDOWN", os.environ.get("AETHER_ANI_COOLDOWN", "300")
            )
        )
        self.allow_skip = bool(
            int(
                os.environ.get(
                    "AETHER_AI_SKIP_ON_HAZARD",
                    os.environ.get("AETHER_ANI_SKIP_ON_HAZARD", "1"),
                )
            )
        )
        self.hist_len = int(os.environ.get("AETHER_AI_HIST", "256"))
        self.action_mask = os.environ.get(
            "AETHER_AI_ACTIONS", "none,ls_down,clip_up,lr_down,softmax_on,sanitize,skip"
        ).split(",")
        self.global_safe_softmax_on = False
        self.global_softmax_patched = False
        self.skip_next_step = False
        self.last_action = "none"
        self.t = 0
        self.window_loss = []
        self.window_finite = []
        self.window_grad = []
        self.arms = [
            "none",
            "ls_down",
            "clip_up",
            "lr_down",
            "softmax_on",
            "sanitize",
            "skip",
        ]
        self.q = {a: 0.0 for a in self.arms}
        self.n = {a: 0 for a in self.arms}
        # lossscale bridge (if ANI scaler exists, prefer it)
        self._ls_get = getattr(self.tr, "_ani_get_loss_scale", None)
        self._ls_set = getattr(self.tr, "_ani_set_loss_scale", None)
        if self._ls_get is None or self._ls_set is None:
            # fallback: keep a local soft-scale multiplier
            self._local_ls = float(os.environ.get("AETHER_ANI_LOSS_SCALE", "1.0"))

            def _get():
                return self._local_ls

            def _set(v):
                self.__dict__.update(_local_ls=float(v))

            self._ls_get, self._ls_set = _get, _set
        # grad clip bridge
        self._set_grad_clip = getattr(self.tr, "_ani_set_grad_clip", None)
        # LR bridge
        self._lr_base = None

        if self.enabled and self.safe_softmax_default:
            self._patch_softmax(True)
        if self.enabled and self.sanitize_opt_default:
            self._sanitize_optimizer_states_safe()

    # ---------- patches ----------
    def _patch_softmax(self, on: bool):
        if on and not self.global_softmax_patched:
            self._orig_softmax = torch.softmax
            self._orig_logsoftmax = F.log_softmax

            def _safe_softmax(x, dim=-1, dtype=None):
                if not torch.is_floating_point(x):
                    return self._orig_softmax(x, dim=dim, dtype=dtype)
                x32 = x.float()
                x32 = x32 - x32.amax(dim=dim, keepdim=True)
                x32 = x32.clamp(min=-50.0, max=50.0)  # strong but safe
                y = torch.exp(x32)
                return (y / (y.sum(dim=dim, keepdim=True) + 1e-12)).to(dtype or x.dtype)

            def _safe_logsoftmax(x, dim=-1, dtype=None):
                if not torch.is_floating_point(x):
                    return F.log_softmax(x, dim=dim, dtype=dtype)
                x32 = x.float()
                x32 = x32 - x32.amax(dim=dim, keepdim=True)
                x32 = x32.clamp(min=-50.0, max=50.0)
                logsumexp = torch.log(torch.exp(x32).sum(dim=dim, keepdim=True) + 1e-12)
                return (x32 - logsumexp).to(dtype or x.dtype)

            torch.softmax = _safe_softmax
            F.log_softmax = _safe_logsoftmax
            self.global_softmax_patched = True
            self.global_safe_softmax_on = True
        elif (not on) and self.global_softmax_patched:
            try:
                torch.softmax = self._orig_softmax
                F.log_softmax = self._orig_logsoftmax
            except Exception:
                pass
            self.global_softmax_patched = False
            self.global_safe_softmax_on = False

    def _sanitize_optimizer_states_safe(self):
        opt = getattr(self.tr, "optimizer", None)
        if opt is None:
            return
        for p in opt.state.values():
            for k, v in list(p.items()):
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    if not torch.isfinite(v).all():
                        p[k] = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)

    # ---------- observation ----------
    def observe_forward(self, loss_tensor, logits=None):
        self.t += 1
        try:
            l = float(loss_tensor.detach().float().clamp(min=-20.0, max=20.0))
        except Exception:
            l = float("inf")
        self.window_loss.append(l)
        if len(self.window_loss) > self.hist_len:
            self.window_loss = self.window_loss[-self.hist_len :]
        finite = 1.0 if math.isfinite(l) else 0.0
        self.window_finite.append(finite)
        if len(self.window_finite) > self.hist_len:
            self.window_finite = self.window_finite[-self.hist_len :]

    def observe_grads(self, model):
        try:
            tot = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if not g.dtype.is_floating_point:
                    continue
                v = float(torch.linalg.norm(g.float()).cpu())
                if math.isfinite(v):
                    tot += v
            self.window_grad.append(tot)
            if len(self.window_grad) > self.hist_len:
                self.window_grad = self.window_grad[-self.hist_len :]
        except Exception:
            pass

    # ---------- policy ----------
    def _ucb_pick(self):
        # restrict by mask
        arms = [a for a in self.arms if a in self.action_mask]
        if not arms:
            arms = ["none"]
        t = max(1, sum(self.n[a] for a in arms))
        best_a, best_val = arms[0], -1e9
        for a in arms:
            q = self.q[a]
            n = max(1, self.n[a])
            u = q + self.ucb_c * math.sqrt(math.log(t + 1.0) / n)
            if u > best_val:
                best_val, best_a = u, a
        return best_a

    def _reward(self):
        # +1 for recent stability, -1 if instability / NaN likely (0 in loss is suspicious too)
        if not self.window_loss:
            return 0.0
        recent = self.window_loss[-min(8, len(self.window_loss)) :]
        finite_recent = self.window_finite[-len(recent) :]
        if sum(finite_recent) < len(recent):
            return -1.0
        if any(abs(x) < 1e-12 for x in recent):
            return -0.5  # suspicious zeros
        # gentle reward for decrease
        if len(recent) >= 4:
            d = (sum(recent[-2:]) - sum(recent[:2])) / 2.0
            return 0.5 if d < 0 else 0.1
        return 0.2

    def decide_and_act(self, step_idx: int):
        if not self.enabled:
            return
        # hazard = too many non-finite in tail or weird zero loss
        hazard = False
        if (
            self.window_finite
            and sum(self.window_finite[-self.hazard_patience :]) < self.hazard_patience
        ):
            hazard = True
        if self.window_loss and any(
            abs(x) < 1e-12 for x in self.window_loss[-self.hazard_patience :]
        ):
            hazard = True
        if not hazard:  # small chance to relax patches
            if (step_idx % max(1, self.cooldown)) == 0:
                self._patch_softmax(self.safe_softmax_default)  # may stay on by default
            # update Q for previous action
            r = self._reward()
            if self.last_action in self.q:
                self.q[self.last_action] = 0.9 * self.q[self.last_action] + 0.1 * r
                self.n[self.last_action] += 1
            return
        # --- hazard path: pick an action ---
        a = self._ucb_pick()
        self.last_action = a
        # apply
        if a == "none":
            return
        if a == "ls_down":
            cur = float(self._ls_get())
            new = max(self.min_ls, cur * self.ls_backoff)
            self._ls_set(new)
        elif a == "clip_up":
            if self._set_grad_clip:
                self._set_grad_clip(self.grad_clip_max)
        elif a == "lr_down" and self.allow_lr_backoff:
            opt = getattr(self.tr, "optimizer", None)
            if opt is not None:
                for g in opt.param_groups:
                    base = g.get("_base_lr", g["lr"])
                    g["_base_lr"] = base
                    g["lr"] = max(1e-7, float(base) * self.lr_backoff)
        elif a == "softmax_on":
            self._patch_softmax(True)
        elif a == "sanitize":
            self._sanitize_optimizer_states_safe()
        elif a == "skip" and self.allow_skip:
            self.skip_next_step = True
        # update stats for arm
        self.n[a] += 1

    # called by loop to check skip
    def should_skip(self):
        if self.skip_next_step:
            self.skip_next_step = False
            return True
        return False

    # ---------- flags & planning ----------
    def set_flag(self, name: str, value: bool):
        if not hasattr(self, "_flags"):
            self._flags = {}
        self._flags[name] = bool(value)

    def get_flag(self, name: str, default=False):
        return bool(getattr(self, "_flags", {}).get(name, default))

    def plan_pre_forward(self):
        # Decide pre-forward toggles from recent stability (e.g., fp32 logits)
        plan = {}
        recent_instab = (
            len(self.window_finite) >= max(2, self.hazard_patience)
            and sum(self.window_finite[-self.hazard_patience :]) < self.hazard_patience
        )
        if recent_instab:
            plan["fp32_logits"] = True
        if bool(int(os.environ.get("AETHER_FP32_LOGITS", "0"))):
            plan["fp32_logits"] = True
        return plan

    def post_forward_assess(self, logits, subx=None, suby=None, pad_id=None):
        # Inspect logits to catch early saturation before backward/step.
        out = {"hazard": False}
        try:
            Lmax = float(logits.detach().float().abs().amax().cpu())
        except Exception:
            Lmax = 0.0
        if Lmax > 75.0:  # conservative threshold for saturation risk
            self.set_flag("fp32_logits", True)
            out["hazard"] = True
            if self.allow_skip:
                self.skip_next_step = True
        try:
            if suby is not None and pad_id is not None:
                mask = suby != pad_id
                if int(mask.sum().item()) <= 1:
                    out["hazard"] = True
                    if self.allow_skip:
                        self.skip_next_step = True
        except Exception:
            pass
        return out

    def sanitize_gradients(self, model):
        # Nan->num for grads; light clamp to avoid wild spikes.
        try:
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad
                if not g.dtype.is_floating_point:
                    continue
                g.data = torch.nan_to_num(g.data, nan=0.0, posinf=1e4, neginf=-1e4)
        except Exception:
            pass


import spiral_chronostasis_v6_3 as chrono

chrono.install_defaults()  # ~/.spiral/chronostasis.json を生成
# === k-bridge (optional) ======================================================
try:
    from kbridge.k_autograd import khuber_loss
    from kbridge.metrics import roc_auc_binary, pr_auc_binary
    from kbridge.kmetrics import (
        ece_and_hist_k,
        ndcg_at_k_seq_k,
        ece_multi_groups_k,
        ece_and_hist_k_bins,
    )
    from kbridge.preproc import byte_normalize_utf8

    _KBRIDGE_AVAILABLE = True
except Exception:
    _KBRIDGE_AVAILABLE = False


# === Class grouping helpers (ByteTokenizer-aware; fallback-safe) =============
def _build_classmap(vocab_size: int, scheme: str = "byte-basic", json_path: str = ""):
    """
    Returns: (classmap_np[int32, shape=(vocab_size,)], group_names[list[str]])
    scheme: "byte-basic" | "byte-compact" | "custom-json"
    - ByteTokenizer: PAD=0, BOS=1, EOS=2, bytes: id=3..258 => b=0..255
    Groups (byte-basic):
      0=control(<32,127 except whitespace), 1=whitespace, 2=digit, 3=alpha,
      4=punct, 5=ascii-other, 6=non-ascii(>=128), 7=special(PAD/BOS/EOS/others)
    """
    names = [
        "control",
        "whitespace",
        "digit",
        "alpha",
        "punct",
        "asciiOther",
        "nonASCII",
        "special",
    ]
    m = _np.full((vocab_size,), 7, dtype=_np.int32)  # default= special/other
    # specials
    for t in [0, 1, 2]:
        if t < vocab_size:
            m[t] = 7
    if scheme == "custom-json" and json_path:
        try:
            import json
            import os

            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        k = int(k)
                        g = int(v)
                        if 0 <= k < vocab_size:
                            m[k] = g
                elif isinstance(obj, list):
                    arr = _np.asarray(obj, dtype=_np.int32).ravel()
                    m[: min(vocab_size, arr.size)] = arr[: min(vocab_size, arr.size)]
                return m, names
        except Exception:
            pass
    # byte-based
    for tid in range(3, min(vocab_size, 259)):
        b = tid - 3
        if b < 32 or b == 127:  # control
            if b in (9, 10, 11, 12, 13):
                m[tid] = 1  # whitespace
            else:
                m[tid] = 0
        elif b == 32 or b in (9, 10, 11, 12, 13):
            m[tid] = 1  # whitespace
        elif 48 <= b <= 57:
            m[tid] = 2  # digit
        elif 65 <= b <= 90 or 97 <= b <= 122:
            m[tid] = 3  # alpha
        elif b <= 127:
            m[tid] = 4 if chr(b) in string.punctuation else 5
        else:
            m[tid] = 6  # non-ASCII
    if scheme == "byte-compact":
        # merge some buckets: asciiOther->punct, control->whitespace
        m[_np.where(m == 5)] = 4
        m[_np.where(m == 0)] = 1
        names = ["ws", "digit", "alpha", "punct", "nonASCII", "special"]  # compact view
        # remap indices to 0..5
        # mapping: ws(1)->0, digit(2)->1, alpha(3)->2, punct(4 or 5)->3, nonASCII(6)->4, special(7)->5
        remap = _np.array([3, 0, 1, 2, 3, 3, 4, 5], dtype=_np.int32)
        m = remap[m]
    return m, names


import string
import spiral_chronostasis_v6_3 as chrono

print(chrono.stats())
# === ultramem (optional drop-in) ===
try:
    import ultramem_patch as up
except Exception:
    up = None
# ========= Optional: external wasm bridge (safe import) =========
try:
    from poly_core_wasm_v12e_wasiX import WasmSIMDBridge  # noqa: F401
except Exception:
    WasmSIMDBridge = None
# ---- 外部ポンプ -------------------------------------------------------------
try:
    from spiral_pump_multi import SpiralPump as SpiralPumpEngine
    from spiral_pump_multi import detect_backend as pump_detect_backend
except Exception:
    # Fallback: allow training/inference without external engine
    SpiralPumpEngine = None

    def pump_detect_backend():
        try:
            import torch
        except Exception:
            return "cpu"

        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


#
# ====== Gradient Checkpointing (MPS-safe) ====================================
def enable_gradient_checkpointing(model, every: int = 1):
    """
    Wrap each N-th TransformerBlock.forward with torch.utils.checkpoint.
    Works on MPS (use_reentrant=False).
    """
    try:
        import torch.utils.checkpoint as ckpt
    except Exception:
        print("[CKPT] torch.utils.checkpoint not available; skipping")
        return
    # すでに適用済みなら何もしない
    for i, blk in enumerate(getattr(model, "blocks", [])):
        if (i + 1) % max(1, int(every)) != 0:
            continue
        if hasattr(blk, "_orig_forward"):
            continue
        blk._orig_forward = blk.forward  # keep original

        def _wrap(b):
            def _fw(self, x, pad_mask=None, **kwargs):
                pm = pad_mask if pad_mask is not None else kwargs.pop("pad_mask", None)

                def inner(_x):
                    if hasattr(b, "_orig_forward"):
                        return b._orig_forward(_x, pad_mask=pm, **kwargs)
                    return b.forward(_x, pad_mask=pm, **kwargs)

                return ckpt.checkpoint(
                    inner, x, use_reentrant=False, preserve_rng_state=False
                )

            return _fw

        blk.forward = types.MethodType(_wrap(blk), blk)
    print(f"[CKPT] enabled (every={every})")


def disable_gradient_checkpointing(model):
    """Restore original forward if wrapped."""
    for blk in getattr(model, "blocks", []):
        if hasattr(blk, "_orig_forward"):
            blk.forward = blk._orig_forward
            delattr(blk, "_orig_forward")
    print("[CKPT] disabled")


# --- misc utils --------------------------------------------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- numerics guard ----------------------------------------------------------
def zero_nan_(t: torch.Tensor):
    if not torch.isfinite(t).all():
        t.data = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


# ====== Tiled SDPA (replace F.sdpa) =========================================
_ORIG_SDPA = F.scaled_dot_product_attention
_DEFAULT_TILE_Q = int(os.environ.get("AETHER_MPS_FLASH_TILE_Q", "192"))
_DEFAULT_TILE_K = int(os.environ.get("AETHER_MPS_FLASH_TILE_K", "320"))
_DEFAULT_FLASH_FP32 = os.environ.get("AETHER_MPS_FLASH_FP32", "1") != "0"
_DEFAULT_WINDOW = int(os.environ.get("AETHER_MPS_FLASH_WINDOW", "0"))
_DEFAULT_GLOBAL_TOKENS = int(os.environ.get("AETHER_MPS_FLASH_GLOBALS", "0"))
_DEFAULT_GLOBAL_STRIDE = int(os.environ.get("AETHER_MPS_FLASH_STRIDE", "0"))

_PATCHED_SDPA = {
    "on": False,
    "tile_q": max(16, _DEFAULT_TILE_Q),
    "tile_k": max(32, _DEFAULT_TILE_K),
    "fp32": bool(_DEFAULT_FLASH_FP32),
    "window": max(0, _DEFAULT_WINDOW),  # 0=OFF, >0: local band (past window only)
    "global_tokens": max(0, _DEFAULT_GLOBAL_TOKENS),  # always-allowed keys from head
    "global_stride": max(0, _DEFAULT_GLOBAL_STRIDE),  # evenly spaced globals (0=off)
}


def _ensure_bhtd(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        return x
    if x.shape[1] < 8 and x.shape[2] > 8:  # (B,T,H,D) -> (B,H,T,D)
        return x.permute(0, 2, 1, 3).contiguous()
    return x


def _build_global_mask(T: int, g0: int, gstep: int, device, dtype=torch.bool):
    if T <= 0 or (g0 <= 0 and gstep <= 0):
        return torch.zeros(T, dtype=dtype, device=device)
    m = torch.zeros(T, dtype=dtype, device=device)
    if g0 > 0:
        m[: min(T, g0)] = True
    if gstep and gstep > 0:
        m[::gstep] = True
    return m


def streaming_sdpa_mps(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    tile_q=192,
    tile_k=320,
    compute_in_fp32=True,
    window_size: int = 0,
    global_tokens: int = 0,
    global_stride: int = 0,
):
    q = _ensure_bhtd(q)
    k = _ensure_bhtd(k)
    v = _ensure_bhtd(v)
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    dtype_in = q.dtype
    calc = (
        torch.float32
        if (compute_in_fp32 and dtype_in in (torch.float16, torch.bfloat16))
        else dtype_in
    )
    q = q.to(calc)
    k = k.to(calc)
    v = v.to(calc)
    out = torch.empty((B, H, Tq, D), dtype=calc, device=q.device)
    scale = 1.0 / math.sqrt(max(1, D))

    def slicemask(qs, qe, ks, ke):
        if attn_mask is None:
            return None
        m = attn_mask
        try:
            return m[(..., slice(qs, qe), slice(ks, ke))]
        except Exception:
            while m.dim() < 4:
                m = m.unsqueeze(0)
            return m.expand(B, 1, qe - qs, ke - ks)

    gmask_full = _build_global_mask(
        Tk, int(global_tokens), int(global_stride), device=q.device
    )

    for qs in range(0, Tq, tile_q):
        qe = min(Tq, qs + tile_q)
        q_blk = q[:, :, qs:qe, :] * scale
        y = torch.zeros((B, H, qe - qs, D), dtype=calc, device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), dtype=calc, device=q.device)
        m = torch.full((B, H, qe - qs, 1), -float("inf"), dtype=calc, device=q.device)
        k_limit = qe if is_causal else Tk

        for ks in range(0, k_limit, tile_k):
            ke = min(k_limit, ks + tile_k)
            k_blk = k[:, :, ks:ke, :]
            v_blk = v[:, :, ks:ke, :]
            s = torch.einsum("bhtd,bhkd->bhtk", q_blk, k_blk)

            ms = slicemask(qs, qe, ks, ke)
            local_mask = None

            if is_causal or window_size > 0:
                q_idx = torch.arange(qs, qe, device=q.device)
                k_idx = torch.arange(ks, ke, device=q.device)
                causal = (
                    (k_idx.unsqueeze(0) > q_idx.unsqueeze(1)) if is_causal else None
                )
                too_far = (
                    ((q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) > int(window_size))
                    if window_size > 0
                    else None
                )
                if causal is not None and too_far is not None:
                    local_mask = causal | too_far
                elif causal is not None:
                    local_mask = causal
                elif too_far is not None:
                    local_mask = too_far
                if local_mask is not None:
                    gsub = gmask_full[ks:ke].unsqueeze(0)
                    local_mask = local_mask & (~gsub)
                    local_mask = local_mask.unsqueeze(0).unsqueeze(0)

            if ms is not None and local_mask is not None:
                ms = ms | local_mask
            elif local_mask is not None:
                ms = local_mask

            if ms is not None:
                s = s.masked_fill(ms, -float("inf"))

            m_ij = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            p = torch.exp(s - m_ij)
            if dropout_p and dropout_p > 0:
                p = F.dropout(p, p=float(dropout_p), training=True)
            y = y * torch.exp(m - m_ij) + torch.einsum("bhtk,bhkd->bhtd", p, v_blk)
            l = l * torch.exp(m - m_ij) + p.sum(dim=-1, keepdim=True)
            m = m_ij

        out[:, :, qs:qe, :] = y / torch.clamp_min(l, 1e-9)

    return out.to(dtype_in)


def _patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return streaming_sdpa_mps(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        _PATCHED_SDPA["tile_q"],
        _PATCHED_SDPA["tile_k"],
        _PATCHED_SDPA["fp32"],
        _PATCHED_SDPA["window"],
        _PATCHED_SDPA["global_tokens"],
        _PATCHED_SDPA["global_stride"],
    )


def enable_tiled_sdpa(tile_q=192, tile_k=320, compute_in_fp32=True):
    if not _PATCHED_SDPA["on"]:
        setattr(F, "scaled_dot_product_attention", _patched_sdpa)
        _PATCHED_SDPA["on"] = True
    _PATCHED_SDPA.update(
        {"tile_q": int(tile_q), "tile_k": int(tile_k), "fp32": bool(compute_in_fp32)}
    )


def set_sliding_window(window: int = 0, global_tokens: int = 0, global_stride: int = 0):
    _PATCHED_SDPA["window"] = max(0, int(window))
    _PATCHED_SDPA["global_tokens"] = max(0, int(global_tokens))
    _PATCHED_SDPA["global_stride"] = max(0, int(global_stride))


def disable_tiled_sdpa():
    if _PATCHED_SDPA["on"]:
        setattr(F, "scaled_dot_product_attention", _ORIG_SDPA)
        _PATCHED_SDPA["on"] = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        print(f"[ENV] invalid int for {name}={raw!r}; using {default}")
        return int(default)


def _auto_enable_mps_flash_attention():
    if os.environ.get("AETHER_DISABLE_MPS_FLASH", "0") == "1":
        return
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return
    except Exception:
        return

    tile_q = max(16, _env_int("AETHER_MPS_FLASH_TILE_Q", _PATCHED_SDPA["tile_q"]))
    tile_k = max(32, _env_int("AETHER_MPS_FLASH_TILE_K", _PATCHED_SDPA["tile_k"]))
    fp32 = os.environ.get("AETHER_MPS_FLASH_FP32", "1") != "0"
    enable_tiled_sdpa(tile_q=tile_q, tile_k=tile_k, compute_in_fp32=fp32)
    window = _env_int("AETHER_MPS_FLASH_WINDOW", _PATCHED_SDPA["window"])
    globals_n = _env_int("AETHER_MPS_FLASH_GLOBALS", _PATCHED_SDPA["global_tokens"])
    stride = _env_int("AETHER_MPS_FLASH_STRIDE", _PATCHED_SDPA["global_stride"])
    if window > 0 or globals_n > 0 or stride > 0:
        set_sliding_window(window, globals_n, stride)
    print(
        f"[MPS][FLASH] enabled streaming attention (tile_q={tile_q}, tile_k={tile_k}, window={window}, globals={globals_n}, stride={stride})"
    )


_auto_enable_mps_flash_attention()


# ====== INT8 base + LoRA (custom; PEFT排他側で利用) ==========================
def _per_channel_symmetric_quant(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # Fallback if weight is not 2D or in_features is zero
        if w.ndim != 2 or w.shape[1] == 0:
            out = w.shape[0] if w.ndim >= 1 else 1
            maxv = w.abs().amax() if w.numel() > 0 else w.new_tensor(1.0)
            scale_val = (maxv / 127.0).to(torch.float32).clamp_min(1e-6)
            wq = torch.zeros_like(w, dtype=torch.int8)
            s = torch.full(
                (out,), float(scale_val.item()), dtype=torch.float32, device=w.device
            )
            return wq, s
        maxv = w.abs().amax(dim=1) + 1e-8
        scale = (maxv / 127.0).to(torch.float32)
        wq = torch.clamp((w / scale.unsqueeze(1)).round_(), -127, 127).to(torch.int8)
        return wq, scale


class LinearInt8Base(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.w_q = nn.Parameter(
            torch.empty(out_f, in_f, dtype=torch.int8), requires_grad=False
        )
        self.s = nn.Parameter(
            torch.ones(out_f, dtype=torch.float32), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.zeros_(self.w_q)
        nn.init.ones_(self.s)

    def load_from_float(self, w: torch.Tensor, b=None):
        with torch.no_grad():
            wq, s = _per_channel_symmetric_quant(w.to(torch.float32))
            self.w_q.copy_(wq)
            self.s.copy_(s)
            if self.b is not None and b is not None:
                self.b.copy_(b)

    def forward(self, x):
        w = self.w_q.to(torch.float32) * self.s.unsqueeze(1)
        y = F.linear(x, w, bias=self.b)
        return y


class LinearInt8LoRA(nn.Module):
    def __init__(self, in_f, out_f, r=160, alpha=320, dropout=0.0, bias=False):
        super().__init__()
        self.base = LinearInt8Base(in_f, out_f, bias=bias)
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        self.scal = float(alpha / max(1, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        z = self.lora_B(self.drop(self.lora_A(x))) * self.scal
        return y + z


def convert_linear_to_int8_lora(
    model: nn.Module,
    r: int = 160,
    alpha: int = 320,
    dropout: float = 0.0,
    include_names: Optional[List[str]] = None,
    exclude_names: Tuple[str, ...] = ("emb", "head"),
    skip_if_out_equals: Optional[int] = None,
) -> int:
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear):
            if getattr(m, "in_features", 1) == 0 or getattr(m, "out_features", 1) == 0:
                print(
                    f"[INT8-LoRA] skip zero-sized Linear: {name} in={getattr(m, 'in_features', None)} out={getattr(m, 'out_features', None)}"
                )
                continue
            if include_names is not None and not any(
                (inc in name) for inc in include_names
            ):
                continue
            if any((exc in name) for exc in exclude_names):
                continue
            p = list(m.parameters())
            w = p[0].detach().to(torch.float32)
            b = p[1].detach() if (len(p) > 1 and p[1] is not None) else None
            q = LinearInt8LoRA(
                m.in_features, m.out_features, r, alpha, dropout, bias=(b is not None)
            )
            q.base.load_from_float(w, b)
            # replace in parent
            parent = model
            path = name.split(".")
            for seg in path[:-1]:
                parent = getattr(parent, seg)
            setattr(parent, path[-1], q)
            n += 1
    return n


# ====== Tokenizer (Byte-level; PAD/BOS/EOS) ==================================
class ByteTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size

    def encode(self, s: str) -> List[int]:
        b = s.encode("utf-8", errors="ignore")[: self.vocab_size - 3]
        return [self.BOS] + [int(x) + 3 for x in b] + [self.EOS]

    def decode(self, ids: List[int]) -> str:
        b = [max(0, min(255, i - 3)) for i in ids if i >= 3]
        return bytes(b).decode("utf-8", errors="ignore")


# ====== RMSNorm / SwiGLU / Rotary ===========================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.g


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4.0):
        super().__init__()
        h = int(d * mult)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base_theta: float = 10_000.0, scaling: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.theta = float(base_theta)
        self.scaling = float(scaling)

    def _freqs(self, T: int, device, dtype):
        dim = self.dim
        inv = 1.0 / (
            self.theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(T, device=device, dtype=torch.float32) * float(self.scaling)
        freqs = torch.einsum("i,j->ij", t, inv)
        return torch.cat([freqs, freqs], dim=-1)

    # --- after ---
    def apply_rotary(self, x):
        """
        x: [B, H, T, D] を想定。最後の次元 D を前半/後半に分け、半次元で回転をかける。
        D が奇数でもパディング→適用→元に戻すので安全。
        """
        import torch
        import torch.nn.functional as F

        B, H, T, D = x.shape
        half = D // 2
        odd = D % 2 == 1
        if odd:
            # 奇数 head_dim の場合は末尾に 1 を詰めてから処理
            x = F.pad(x, (0, 1))
            D += 1
            half = D // 2

        # 周波数は必ず「半次元」で作る（ここが要点！）
        device, dtype = x.device, x.dtype
        # self.theta は rope_theta、self.scaling は dict か None を想定
        inv = 1.0 / (
            self.theta
            ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        t = torch.arange(T, device=device, dtype=torch.float32)

        # freqs: [T, half]
        freqs = torch.einsum("t,d->td", t, inv)

        # rope_scaling（linear 等）を使っている場合のみスケール
        if isinstance(self.scaling, dict):
            sc = self.scaling.get("factor") or self.scaling.get("scale")
            if sc:
                freqs = freqs / float(sc)

        # cos/sin: [1, 1, T, half] に整形（H にブロードキャスト）
        cos = torch.cos(freqs).to(dtype).view(1, 1, T, half)
        sin = torch.sin(freqs).to(dtype).view(1, 1, T, half)

        # 前半/後半に分解して回転
        x1 = x[..., :half]
        x2 = x[..., half : half * 2]
        xr = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # 奇数 head_dim の場合は詰めた 1 を落とす
        if odd:
            xr = xr[..., : D - 1]

        return xr


# ====== MHA / Block / Model ==================================================
class MHA(nn.Module):
    """single qkv proj + GQA/MQA kv_heads"""

    def __init__(
        self,
        d,
        heads,
        dropout=0.0,
        use_rope: bool = False,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.hq = int(heads)
        self.hk = int(kv_heads) if kv_heads else int(heads)
        assert self.hq % self.hk == 0, "hq must be divisible by hk"
        self.d = int(d)
        self.dk = d // self.hq
        self.drop = nn.Dropout(dropout)
        self.qkv = nn.Linear(d, self.hq * self.dk + 2 * self.hk * self.dk, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.use_rope = bool(use_rope)
        self.rope = (
            RotaryEmbedding(self.dk, base_theta=rope_theta, scaling=rope_scaling)
            if self.use_rope
            else None
        )

    def forward(self, x, pad_mask=None, attn_mask=None, is_causal=True):
        B, T, D = x.shape
        z = self.qkv(x)
        q_end = self.hq * self.dk
        kv_end = q_end + 2 * self.hk * self.dk
        q = z[:, :, :q_end].view(B, T, self.hq, self.dk).permute(0, 2, 1, 3)
        kv = (
            z[:, :, q_end:kv_end].view(B, T, self.hk, 2, self.dk).permute(0, 2, 1, 3, 4)
        )
        k = kv[:, :, :, 0, :]
        v = kv[:, :, :, 1, :]
        if self.hk != self.hq:
            rep = self.hq // self.hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if self.use_rope:
            if self.rope is None:
                raise RuntimeError("rope not initialized")
            self.rope.to(x.device, x.dtype)
            if (self.dk % 2) != 0:
                q = F.pad(q, (0, 1))
                k = F.pad(k, (0, 1))
            q = self.rope.apply_rotary(q)
            k = self.rope.apply_rotary(k)
            if (self.dk % 2) != 0:
                q = q[..., : self.dk]
                k = k[..., : self.dk]

        m = None
        if pad_mask is not None:
            m = (~pad_mask).unsqueeze(1).unsqueeze(2).expand(B, 1, T, T)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=m if attn_mask is None else attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d,
        heads,
        dropout=0.0,
        *,
        ff_mult: float = 2.6666667,
        use_rope: bool = False,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = MHA(
            d,
            heads,
            dropout,
            use_rope=use_rope,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            kv_heads=kv_heads,
        )
        self.n2 = RMSNorm(d)
        self.ff = SwiGLU(d, mult=ff_mult)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.n1(x), pad_mask=pad_mask))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class AetherPumpSimple(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        dropout=0.1,
        max_len=4096,
        pad_id=0,
        tie_weights=True,
        ff_mult: float = 2.6666667,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        use_abs_pos: bool = False,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.use_abs_pos = bool(use_abs_pos)
        if self.use_abs_pos:
            self.pos = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout,
                    ff_mult=ff_mult,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    kv_heads=kv_heads,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.emb.weight
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device
        pad_mask = input_ids != self.pad_id
        x = self.emb(input_ids)
        if self.use_abs_pos:
            x = x + self.pos[:T, :].to(device)
        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask)
        x = self.ln_f(x)
        return self.head(x)

    def forward_with_bias(
        self, input_ids: torch.Tensor, bias: torch.Tensor, attention_mask=None, **kwargs
    ):
        B, T = input_ids.shape
        device = input_ids.device
        pad_mask = input_ids != self.pad_id
        x = self.emb(input_ids) + bias.unsqueeze(1)
        if self.use_abs_pos:
            x = x + self.pos[:T, :].to(device)
        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def set_rope_params(
        self,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        max_pos: Optional[int] = None,
    ):
        mp = max_pos if max_pos is not None else getattr(self, "max_len", 4096)
        for blk in self.blocks:
            if hasattr(blk, "attn") and isinstance(blk.attn, MHA):
                blk.attn.use_rope = bool(use_rope)
                if blk.attn.use_rope:
                    rope_dim = (
                        blk.attn.dk if (blk.attn.dk % 2) == 0 else (blk.attn.dk + 1)
                    )
                    blk.attn.rope = RotaryEmbedding(
                        rope_dim,
                        base_theta=float(rope_theta),
                        scaling=float(rope_scaling),
                    )


# ====== Collate / Datasets ===================================================
def collate_lm_safe(batch, pad_id: int):
    # batch: List[List[int]]  or  List[Tuple[List[int], List[int]]]
    if (
        len(batch) > 0
        and isinstance(batch[0], (tuple, list))
        and isinstance(batch[0][0], list)
    ):
        seqs = [b[0] for b in batch]  # 入力側のみ使用
    else:
        seqs = batch
    mx = max(2, max(len(x) for x in seqs))
    X = torch.full((len(seqs), mx), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        X[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return X[:, : mx - 1], X[:, 1:mx]


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        glob_pat: str,
        tok: ByteTokenizer,
        pack_len: int = 1024,
        buffer_size: int = 8192,
        infinite: bool = True,
        seed: int = 1337,
    ):
        super().__init__()
        self.glob = glob_pat
        self.tok = tok
        self.pack_len = pack_len
        self.buffer_size = buffer_size
        self.infinite = infinite
        self.seed = seed

    def _files(self):
        # 再帰で拾う。0件なら即エラーで可視化（無限待ち防止）
        fs = sorted(glob.glob(self.glob, recursive=True))
        if not fs:
            raise FileNotFoundError(
                f"[DATA] No files match: {self.glob}  (cwd={os.getcwd()})"
            )
        random.Random(self.seed).shuffle(fs)
        return fs

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            for f in self._files():
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        buf = ""
                        for line in fh:
                            buf += line
                            if len(buf) >= self.buffer_size:
                                if (
                                    _KBRIDGE_AVAILABLE
                                    and os.environ.get("KBRIDGE_PREPROC", "0") == "1"
                                ):
                                    try:
                                        nbytes = byte_normalize_utf8(buf)
                                        buf = nbytes.decode("utf-8", errors="ignore")
                                    except Exception:
                                        pass
                                ids = self.tok.encode(buf)
                                buf = ""
                                for s in range(0, max(1, len(ids) - 1), self.pack_len):
                                    block = ids[s : s + self.pack_len]
                                    yield block, block
                        if buf:
                            if (
                                _KBRIDGE_AVAILABLE
                                and os.environ.get("KBRIDGE_PREPROC", "0") == "1"
                            ):
                                try:
                                    nbytes = byte_normalize_utf8(buf)
                                    buf = nbytes.decode("utf-8", errors="ignore")
                                except Exception:
                                    pass
                            ids = self.tok.encode(buf)
                            for s in range(0, max(1, len(ids) - 1), self.pack_len):
                                block = ids[s : s + self.pack_len]
                                yield block, block
                except Exception:
                    pass
            if not self.infinite:
                break


# ====== Trainer utils ========================================================
@dataclass
class CurriculumStage:
    until_step: int
    max_len: int


@dataclass
class TrainConfig:
    epochs: int = 1
    max_steps: Optional[int] = 2000
    safetensor_every: int = 0
    batch_size: int = 1
    micro_batch: int = 1
    lr: float = 2e-4
    warmup_steps: int = 300
    grad_clip: Optional[float] = 1.0
    max_len: int = 2048
    out_dir: str = "runs/v28"
    seed: int = 1337
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 500
    use_tb: bool = False
    curriculum: List[CurriculumStage] = field(default_factory=list)
    token_dropout: float = 0.01
    byte_noise: float = 0.0
    span_mask_prob: float = 0.02
    span_mask_len: int = 8
    tiled_q: int = 192
    tiled_k: int = 320
    mps_sync_every: int = 0
    # LVI regs / switches
    lvi_mv_weight: float = 0.10
    lvi_two_view_weight: float = 0.05
    lvi_enable: bool = False
    lvi_k: int = 64
    lvi_alpha_mode: str = "sparsemax"
    lvi_every: int = 4
    # Attention runtime
    window_size: int = 0
    global_tokens: int = 0
    global_stride: int = 0
    # Optimizer runtime
    opt_cpu8bit: bool = False
    # Intention loss
    intent_weight: float = 0.0
    intent_margin: float = 0.10
    intent_sample_frac: float = 0.25
    intent_every: int = 4
    # ReLoRA
    relora_every: int = 0

    # GaLore-like optimizer
    opt_galore: bool = False
    galore_rank: int = 64
    # GQA
    kv_heads: Optional[int] = None
    # M4 / MPS runtime controls
    prefer_bfloat16: bool = True
    matmul_precision: str = "high"
    compile: bool = False
    compile_backend: str = "aot_eager"
    compile_dynamic: bool = False
    adaptive_microbatch: bool = True
    adaptive_micro_max: int = 16
    adaptive_micro_recover: int = 256
    oom_retries: int = 3
    grad_scaler: bool = True
    empty_cache_every: int = 0
    # Data pipeline / host→device controls
    loader_num_workers: int = 0
    loader_prefetch_factor: int = 2
    loader_persistent_workers: bool = False
    loader_pin_memory: bool = False
    prefetch_to_device: bool = True
    disallow_mps_fallback: bool = True


class FabricLite:
    def __init__(self, outdir, use_tb=False):
        self.outdir = outdir
        self.use_tb = use_tb
        self.tb = None
        try:
            if use_tb:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=outdir)
        except Exception:
            self.tb = None

    def log(self, d: Dict[str, float], step: int = None):
        if self.tb:
            for k, v in d.items():
                try:
                    self.tb.add_scalar(k, float(v), global_step=step)
                except Exception:
                    pass
        ks = ", ".join([f"{k}={v:.4f}" for k, v in d.items()])
        print(f"[LOG] step={step} | {ks}")


class ThroughputMeter:
    def __init__(self, beta: float = 0.90):
        self.beta = float(beta)
        self.value = 0.0
        self.ready = False

    def update(self, tokens: float, seconds: float) -> float:
        if seconds <= 0:
            return self.value if self.ready else 0.0
        inst = float(tokens) / max(seconds, 1e-6)
        if not self.ready:
            self.value = inst
            self.ready = True
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * inst
        return self.value

    def reset(self):
        self.value = 0.0
        self.ready = False


class ChronoScheduler:
    def __init__(self, opt: torch.optim.Optimizer, cfg: TrainConfig, total_steps: int):
        self.opt = opt
        self.cfg = cfg
        self.total = total_steps
        self.step_id = 0
        self.base_lr = cfg.lr
        self.warm = cfg.warmup_steps

    def _setlr(self, mult: float):
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * mult

    def step(self):
        self.step_id += 1
        s = self.step_id
        if s <= self.warm:
            m = s / max(1, self.warm)
        else:
            t = (s - self.warm) / max(1, self.total - self.warm)
            m = 0.5 * (1 + math.cos(math.pi * t))
        self._setlr(m)
        return m


# ====== Augment (light) ======================================================
class PsyAugment:
    def __init__(self, tok: ByteTokenizer):
        self.tok = tok
        self.token_dropout = 0.0
        self.byte_noise = 0.0
        self.span_mask_prob = 0.0
        self.span_len = 8

    def update(self, token_dropout=0.0, byte_noise=0.0, span_mask_prob=0.0, span_len=8):
        self.token_dropout = float(token_dropout)
        self.byte_noise = float(byte_noise)
        self.span_mask_prob = float(span_mask_prob)
        self.span_len = int(span_len)

    @torch.no_grad()
    def apply_input(self, x: torch.Tensor, pad_id: int):
        if (
            self.token_dropout <= 0
            and self.byte_noise <= 0
            and self.span_mask_prob <= 0
        ):
            return x
        B, T = x.shape
        x = x.clone()
        if self.token_dropout > 0:
            m = torch.rand_like(x.to(torch.float32)) < self.token_dropout
            x.masked_fill_(m.to(torch.bool), pad_id)
        # byte_noise / span_mask は必要なら追加
        return x


# ====== PEFT: LoRA (optional) ===============================================
PEFT_AVAILABLE = False
try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType

    PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    PeftModel = None  # type: ignore


def apply_peft_lora(
    model: nn.Module,
    r: int = 160,
    alpha: int = 320,
    dropout: float = 0.05,
    targets: Optional[List[str]] = None,
) -> nn.Module:
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft not installed. `pip install peft`")
    if targets is None:
        targets = ["qkv", "proj"]  # unified projection
    conf = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, conf)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def save_peft_adapter_if_any(model: nn.Module, out_dir: str):
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        print(f"[PEFT] saved adapter to: {out_dir}")


def merge_peft_for_inference(model: nn.Module) -> nn.Module:
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        model = model.merge_and_unload()
        print("[PEFT] merged and unloaded adapters.")
    return model


def apply_hybrid_lora(
    model: nn.Module,
    peft_targets: List[str],
    int8_include: Optional[List[str]],
    r: int = 160,
    alpha: int = 320,
    dropout: float = 0.05,
) -> nn.Module:
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft not installed for hybrid mode")
    model = apply_peft_lora(
        model, r=r, alpha=alpha, dropout=dropout, targets=peft_targets
    )
    if int8_include:
        n = convert_linear_to_int8_lora(
            model.base_model,
            r=r,
            alpha=alpha,
            dropout=0.0,
            include_names=int8_include,
            exclude_names=("emb", "head"),
            skip_if_out_equals=getattr(model.base_model, "vocab_size", None),
        )
        print(f"[HYBRID] INT8+LoRA injected: {n}")
    return model


# ====== CPU AdamW (8bit-ish) ===============================================
class CPUAdamW8(torch.optim.Optimizer):
    """m,v を CPU に保持。v を int8 量子化（対数尺度）して更新コストと常駐RAMを軽量化。"""

    def __init__(
        self,
        params,
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        quantize_v=True,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.quantize_v = bool(quantize_v)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd and wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                g = p.grad.detach()
                if not torch.isfinite(g).all():
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                g_cpu = g.to("cpu", dtype=torch.float16, non_blocking=False)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        g_cpu, memory_format=torch.preserve_format
                    )
                    if self.quantize_v:
                        state["v_q"] = torch.zeros_like(g_cpu, dtype=torch.int8)
                        state["v_s"] = torch.ones((), dtype=torch.float32)
                    else:
                        state["v"] = torch.zeros_like(
                            g_cpu, memory_format=torch.preserve_format
                        )

                m = state["m"]
                if self.quantize_v:
                    v_q = state["v_q"]
                    v_s = state["v_s"]
                    v = v_q.float() * v_s
                else:
                    v = state["v"]

                state["step"] += 1
                t = state["step"]
                m.mul_(beta1).add_(g_cpu, alpha=(1.0 - beta1))
                v.mul_(beta2).addcmul_(g_cpu, g_cpu, value=(1.0 - beta2))

                if self.quantize_v:
                    vmax = v.abs().amax()
                    v_s = (vmax / 127.0).clamp_min(1e-6)
                    state["v_s"] = v_s
                    v_q.copy_((v / v_s).clamp_(-127, 127).round_().to(torch.int8))

                bc1 = 1 - beta1**t
                bc2 = 1 - beta2**t
                mhat = m / bc1
                vhat = v / bc2
                upd = (mhat / (vhat.sqrt() + eps)).to(torch.float16)
                p.data.add_(-lr * upd.to(device=p.data.device, dtype=p.data.dtype))
        return loss


# ====== Teacher / LVI / Intention loss ======================================
def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).clamp(-1.0, 1.0)


class _SimpleTeacher(nn.Module):
    """固定写像: 2-gram ハッシュ → Embedding → Lin(D)"""

    def __init__(self, d_model: int, buckets: int = 65536, proj_dim: int = 256):
        super().__init__()
        self.table = nn.Embedding(buckets, proj_dim)
        with torch.no_grad():
            nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        self.proj = nn.Linear(proj_dim, d_model, bias=False)
        for p in self.parameters():
            p.requires_grad = False
        self.buckets = int(buckets)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        device = ids.device
        h = (
            ids[:, :-1].long() * 1315423911 + ids[:, 1:].long() * 2654435761
        ) % self.buckets
        if h.numel() == 0:
            h = torch.zeros(B, 1, dtype=torch.long, device=device)
        feat = self.table(h).mean(dim=1)
        return self.proj(feat)


class _LVIEngine(nn.Module):
    """軽量 LVI: teacherベクトルを微量 bias として注入"""

    def __init__(self, d_model: int, scale: float = 0.02):
        super().__init__()
        self.teacher = _SimpleTeacher(d_model)
        self.scale = float(scale)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, ids: torch.Tensor):
        vec = self.teacher(ids)  # (B,D)
        bias = vec * self.scale
        logs = {
            "mv_reg": ids.new_tensor(0.0, dtype=torch.float32),
            "two_view": ids.new_tensor(0.0, dtype=torch.float32),
        }
        return bias, logs


def _make_negative_ids(ids: torch.Tensor) -> torch.Tensor:
    if ids.numel() == 0:
        return ids
    neg = ids.clone()
    core = neg[:, 1:].clone()
    if core.size(1) > 1:
        core = torch.roll(core, shifts=1, dims=1)
    neg[:, 1:] = core
    return neg


def _contrastive_intent_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    emb_weight: torch.Tensor,
    pad_id: int,
    t_pos: Optional[torch.Tensor],
    t_neg: Optional[torch.Tensor],
    margin: float = 0.10,
    sample_frac: float = 0.25,
) -> torch.Tensor:
    B, T, V = logits.shape
    if sample_frac < 1.0:
        stride = max(1, int(1.0 / max(1e-6, sample_frac)))
        idx = torch.arange(0, T, device=logits.device, step=stride)
        logits = logits[:, idx, :]
        targets = targets[:, idx]
    mask = (targets != pad_id).float()
    if mask.sum() <= 0:
        return logits.new_tensor(0.0)

    with torch.no_grad():
        maxv = logits.amax(dim=-1, keepdim=True)
        zero_nan_(maxv)
    probs = torch.softmax((logits - maxv).to(torch.float32), dim=-1).clamp_min(1e-6)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    E = emb_weight.to(probs.dtype)
    E_pred = torch.einsum("btv,vd->btd", probs, E)

    if t_pos is not None:
        pos = t_pos.to(E_pred.dtype).unsqueeze(1).expand_as(E_pred)
        cos_pos = _safe_cos(E_pred, pos).mean()
    else:
        gt = targets.view(-1).clamp_min(0)
        E_gt = F.embedding(gt, E).view_as(E_pred)
        cos_pos = _safe_cos(E_pred, E_gt).mean()

    loss = (1.0 - cos_pos).mean()
    if t_neg is not None:
        neg = t_neg.to(E_pred.dtype).unsqueeze(1).expand_as(E_pred)
        cos_neg = _safe_cos(E_pred, neg).mean()
        loss = loss + F.relu(cos_neg - cos_pos + float(margin)).mean()
    return loss


# ====== Trainer (MPS/Fabric) =============================================
class AetherTrainerBase:
    def __init__(self, model: AetherPumpSimple, tok: ByteTokenizer, cfg: TrainConfig):
        self.model = model
        self.tok = tok
        self.cfg = cfg
        self.device = detect_device()

        matmul_mode = getattr(self.cfg, "matmul_precision", None)
        if matmul_mode:
            try:
                torch.set_float32_matmul_precision(str(matmul_mode))
                print(f"[AMP] matmul precision set to {matmul_mode}")
            except Exception as e:
                print("[AMP] matmul precision set failed:", e)

        self.model.to(self.device)

        if getattr(self.cfg, "compile", False):
            backend = getattr(self.cfg, "compile_backend", "aot_eager")
            dynamic = bool(getattr(self.cfg, "compile_dynamic", False))
            try:
                self.model = torch.compile(
                    self.model, backend=str(backend), dynamic=dynamic
                )
                print(
                    f"[COMPILE] torch.compile enabled (backend={backend}, dynamic={dynamic})"
                )
            except Exception as e:
                print("[COMPILE] torch.compile failed:", e)
        enable_tiled_sdpa(cfg.tiled_q, cfg.tiled_k, compute_in_fp32=True)
        try:
            set_sliding_window(
                int(cfg.window_size), int(cfg.global_tokens), int(cfg.global_stride)
            )
        except Exception:
            pass
        self.fabric = FabricLite(cfg.out_dir, use_tb=cfg.use_tb)
        self.psy = PsyAugment(tok)
        self._global_step = 0
        beta_env = os.environ.get("AETHER_TPS_EMA_BETA", "")
        tps_beta = 0.90
        if beta_env:
            try:
                tps_beta = float(beta_env)
            except Exception:
                print(
                    f"[THROUGHPUT] Invalid AETHER_TPS_EMA_BETA={beta_env!r}; using {tps_beta}"
                )
        self._tps_meter = ThroughputMeter(beta=tps_beta)
        self._last_tok_per_sec = 0.0
        self._last_tok_per_sec_ema = 0.0

        wd = 0.01

        # ultramem optimizer factory (if autopatch installed)
        if (up is not None) and hasattr(self.model, "_ultramem_make_optimizer"):
            self.opt = self.model._ultramem_make_optimizer(lr=self.cfg.lr, wd=wd)
            print("[OPT] ultramem optimizer factory used")
        else:
            if getattr(self.cfg, "opt_cpu8bit", False):
                self.opt = CPUAdamW8(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=cfg.lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=wd,
                    quantize_v=True,
                )
                print("[OPT] CPUAdamW8 (v:int8, m:fp16)")
            else:
                self.opt = (
                    GaLoreAdamW(
                        [p for p in self.model.parameters() if p.requires_grad],
                        lr=cfg.lr,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                        weight_decay=wd,
                        rank=int(getattr(cfg, "galore_rank", 64)),
                    )
                    if getattr(cfg, "opt_galore", False)
                    else torch.optim.AdamW(
                        [p for p in self.model.parameters() if p.requires_grad],
                        lr=cfg.lr,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                        weight_decay=wd,
                    )
                )
                print("[OPT] AdamW on MPS")

        if self.device.type == "mps" and getattr(self.cfg, "prefer_bfloat16", True):
            try:
                torch.zeros(1, device=self.device, dtype=torch.bfloat16)
                self.amp_dtype = torch.bfloat16
                print("[AMP] Using bfloat16 autocast on MPS")
            except Exception:
                self.amp_dtype = torch.float16
                print("[AMP] Falling back to float16 autocast")
        else:
            self.amp_dtype = torch.float16

        if not hasattr(self, "amp_dtype"):
            self.amp_dtype = torch.float16

        self._scaler_enabled = self.device.type == "mps" and bool(
            getattr(self.cfg, "grad_scaler", True)
        )
        self.scaler = GradScaler() if self._scaler_enabled else None
        if self.scaler is not None:
            print("[AMP] GradScaler(mps) enabled")

        self._prefetch_to_device = (
            self.device.type == "mps"
            and bool(getattr(self.cfg, "prefetch_to_device", True))
        )
        if self._prefetch_to_device:
            print("[MPS] host batch prefetch enabled")

        if self.device.type == "mps" and bool(
            getattr(self.cfg, "disallow_mps_fallback", True)
        ):
            try:
                import torch.backends.mps as _mps_backend

                _mps_backend.fallback_allow_all(False)
                print("[MPS] CPU fallback disabled (strict MPS kernels)")
            except Exception as e:
                print("[MPS] fallback control unavailable:", e)

        self._teacher = None
        if (self.cfg.intent_weight > 0.0) or bool(
            getattr(self.cfg, "lvi_enable", False)
        ):
            try:
                self._teacher = _SimpleTeacher(getattr(self.model, "d_model", 256)).to(
                    self.device
                )
                print("[INTENT] SimpleTeacher enabled")
            except Exception:
                self._teacher = None

        self._use_lvi = bool(getattr(self.cfg, "lvi_enable", False))
        self._lvi = (
            _LVIEngine(self.model.d_model, scale=0.02).to(self.device)
            if self._use_lvi
            else None
        )
        self._lvi_cache = {"bias": None, "logs": None, "step": -1}
        # ANI-AI controller (opt-in via env)
        self._ai = _AetherNumpyAIController(self)
        # Adaptive micro-batch & memory controls
        self._adaptive_micro = bool(getattr(self.cfg, "adaptive_microbatch", True))
        self._micro_active = max(1, int(getattr(self.cfg, "micro_batch", 1)))
        self._micro_base = self._micro_active
        self._micro_cap = max(
            self._micro_active,
            int(getattr(self.cfg, "adaptive_micro_max", self._micro_active)),
        )
        self._micro_recover_every = max(
            0, int(getattr(self.cfg, "adaptive_micro_recover", 256))
        )
        self._micro_stable = 0
        self._oom_retry_limit = max(0, int(getattr(self.cfg, "oom_retries", 3)))
        self._empty_cache_every = max(0, int(getattr(self.cfg, "empty_cache_every", 0)))
        if self._adaptive_micro and self._micro_cap > self._micro_base:
            print(
                f"[ADAPT] micro_batch base={self._micro_base}, cap={self._micro_cap}"
            )
        # --- k-bridge integration knobs (safe opt-in; default on if installed) ---
        self._k_enabled = bool(
            int(os.environ.get("KBRIDGE_ENABLE", "1" if _KBRIDGE_AVAILABLE else "0"))
        )
        self._k_reg_w = float(os.environ.get("KBRIDGE_REG_W", "0.0"))
        self._kb_buf_cap = int(os.environ.get("KBRIDGE_BUF_CAP", "200000"))
        self._kb_ece_bins = int(os.environ.get("KBRIDGE_ECE_BINS", "15"))
        self._kb_metrics_every = int(
            os.environ.get(
                "KBRIDGE_METRICS_EVERY", str(max(1, getattr(cfg, "log_every", 10)))
            )
        )
        # classwise ECE config & length buckets
        self._kb_ece_by = (
            os.environ.get("KBRIDGE_ECE_BY", "pred").strip().lower()
        )  # "pred" | "true" | "predgrp" | "truegrp"
        self._kb_ece_c_top = int(os.environ.get("KBRIDGE_ECE_C_TOP", "5"))
        self._kb_ece_c_min = int(os.environ.get("KBRIDGE_ECE_C_MIN", "1000"))
        self._kb_ece_c_list = [
            int(x)
            for x in os.environ.get("KBRIDGE_ECE_C_LIST", "").split(",")
            if x.strip().isdigit()
        ]
        _len_edges_env = os.environ.get("KBRIDGE_LEN_BUCKETS", "0,128,512,2048,1000000")
        try:
            _edges = [int(x) for x in _len_edges_env.split(",") if x.strip()]
            _edges = sorted(set([x for x in _edges if x >= 0]))
            if len(_edges) < 2:
                _edges = [0, 1_000_000]
            self._kb_len_edges = _np.asarray(_edges, dtype=_np.int64)
        except Exception:
            self._kb_len_edges = _np.asarray([0, 1_000_000], dtype=_np.int64)
        # token-level buffers
        self._kb_buf_pmax = []
        self._kb_buf_labels = []
        self._kb_buf_predcls = []
        self._kb_buf_truecls = []
        self._kb_buf_lenbin = []
        # sequence-level NDCG buffer
        self._kb_seq_scores = []
        self._kb_seq_gains = []
        # class groups (built from vocab & scheme or custom json)
        self._kb_class_scheme = (
            os.environ.get("KBRIDGE_CLASSMAP", "byte-basic").strip().lower()
        )
        self._kb_class_json = os.environ.get("KBRIDGE_CLASSMAP_FILE", "").strip()
        try:
            self._kb_classmap, self._kb_group_names = _build_classmap(
                vocab_size=getattr(self.model, "vocab_size", 32000),
                scheme=self._kb_class_scheme,
                json_path=self._kb_class_json,
            )
            self._kb_group_count = int(self._kb_classmap.max()) + 1
        except Exception:
            self._kb_classmap = _np.zeros(
                (getattr(self.model, "vocab_size", 32000),), dtype=_np.int32
            )
            self._kb_group_names = []
            self._kb_group_count = int(self._kb_classmap.max()) + 1
        # 2D ECE（len×class）や T sweep/quantile bins
        self._kb_enable_2d = bool(int(os.environ.get("KBRIDGE_ECE_2D", "0")))
        self._kb_2d_max_cells = int(os.environ.get("KBRIDGE_2D_MAX_CELLS", "24"))
        self._kb_tsweep = [
            float(x)
            for x in os.environ.get("KBRIDGE_TSWEEP", "").split(",")
            if x.strip()
        ]

    def _rebuild_optimizer(self):
        wd = 0.01
        if getattr(self.cfg, "opt_cpu8bit", False):
            self.opt = CPUAdamW8(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.cfg.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=wd,
                quantize_v=True,
            )
            print("[OPT] CPUAdamW8 (v:int8, m:fp16)")
        else:
            self.opt = (
                GaLoreAdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.cfg.lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=wd,
                    rank=int(getattr(self.cfg, "galore_rank", 64)),
                )
                if getattr(self.cfg, "opt_galore", False)
                else torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.cfg.lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=wd,
                )
            )
            print("[OPT] AdamW on MPS")

    def _relora_cycle(self):
        if not PEFT_AVAILABLE or not isinstance(self.model, PeftModel):
            return False
        try:
            self.model = merge_peft_for_inference(self.model).to(self.device)
            targets = [
                s.strip()
                for s in str(getattr(self.cfg, "peft_targets", "qkv,proj")).split(",")
                if s.strip()
            ]
            self.model = apply_peft_lora(
                self.model,
                r=int(getattr(self.cfg, "lora_r", 160)),
                alpha=int(getattr(self.cfg, "lora_alpha", 320)),
                dropout=float(getattr(self.cfg, "lora_dropout", 0.05)),
                targets=targets,
            ).to(self.device)
            self._rebuild_optimizer()
            print("[ReLoRA] merged & re-applied adapters; optimizer rebuilt.")
            return True
        except Exception as e:
            print("[ReLoRA] failed:", e)
            return False

    def _make_loader(self, ds, shuffle, max_len_for_collate):
        is_iter = isinstance(ds, IterableDataset)
        num_workers = max(0, int(getattr(self.cfg, "loader_num_workers", 0)))
        persistent = (
            num_workers > 0
            and bool(getattr(self.cfg, "loader_persistent_workers", False))
        )
        pin_memory = bool(getattr(self.cfg, "loader_pin_memory", False))
        loader_kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=(False if is_iter else bool(shuffle)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            collate_fn=lambda b: collate_lm_safe(b, pad_id=self.tok.PAD),
        )
        prefetch_factor = getattr(self.cfg, "loader_prefetch_factor", None)
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        return DataLoader(
            ds,
            **loader_kwargs,
        )

    def _ce_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, pad_id: int
    ) -> torch.Tensor:
        B, T, V = logits.shape
        logits_fp = (
            logits.float() if getattr(self, "_ai_fp32_logits", False) else logits
        )
        loss = F.cross_entropy(
            logits_fp.view(B * T, V),
            targets.view(B * T),
            ignore_index=pad_id,
            reduction="mean",
        )
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_logits(self, ids: torch.Tensor):
        if self._use_lvi and self._lvi is not None:
            cur = self._global_step
            every = max(1, int(getattr(self.cfg, "lvi_every", 1)))
            do_lvi = (
                (every <= 1) or (cur % every == 0) or (self._lvi_cache["bias"] is None)
            )
            if do_lvi:
                bias, logs = self._lvi(ids.to(self.device))
                self._lvi_cache = {"bias": bias.detach(), "logs": logs, "step": cur}
            else:
                B, T = ids.shape
                bias = self._lvi_cache["bias"]
                if bias.size(0) != B:
                    bias = bias[:B]
                logs = self._lvi_cache["logs"]
            logits = self.model.forward_with_bias(
                ids, bias.to(ids.device, dtype=self.model.emb.weight.dtype)
            )
            return logits, logs
        else:
            logits = self.model(ids)
            return logits, {
                "mv_reg": torch.tensor(0.0, device=ids.device),
                "two_view": torch.tensor(0.0, device=ids.device),
            }

    # External hook to feed sequence-level eval (scores/gains per sequence)
    def kb_push_seq_eval(self, scores_row, gains_row):
        try:
            import numpy as _np

            s = _np.asarray(scores_row, dtype=_np.float64).ravel()
            g = _np.asarray(gains_row, dtype=_np.float64).ravel()
            if s.shape == g.shape and s.size > 0:
                self._kb_seq_scores.append(s)
                self._kb_seq_gains.append(g)
        except Exception:
            pass


class AetherTrainerMPS(AetherTrainerBase):
    def __init__(self, *a, ckpt_every: int = 1, **k):
        opt_override = k.pop("opt", None)
        super().__init__(*a, **k)
        if opt_override is not None:
            try:
                self.opt = opt_override
            except Exception:
                self.opt = opt_override
        if ckpt_every > 0:
            enable_gradient_checkpointing(self.model, every=ckpt_every)

    def fit(self, train_ds, val_ds: Optional[Dataset] = None):
        # Curriculum
        curr = (
            self.cfg.curriculum
            if self.cfg.curriculum
            else [
                CurriculumStage(
                    until_step=self.cfg.warmup_steps, max_len=min(self.cfg.max_len, 512)
                ),
                CurriculumStage(
                    until_step=int((self.cfg.max_steps or 10_000) * 0.5),
                    max_len=min(self.cfg.max_len, 1024),
                ),
                CurriculumStage(
                    until_step=(self.cfg.max_steps or 10_000), max_len=self.cfg.max_len
                ),
            ]
        )

        def curr_params(step):
            for st in curr:
                if step <= st.until_step:
                    return {"max_len": st.max_len}
            return {"max_len": curr[-1].max_len}

        # Loaders
        train_dl = self._make_loader(
            train_ds, shuffle=True, max_len_for_collate=curr[0].max_len
        )
        val_dl = (
            self._make_loader(
                val_ds, shuffle=False, max_len_for_collate=curr[0].max_len
            )
            if val_ds is not None
            else None
        )

        try:
            steps_per_epoch = len(train_dl)
        except Exception:
            steps_per_epoch = self.cfg.max_steps or 10**9
        total_steps = self.cfg.max_steps or int(self.cfg.epochs * steps_per_epoch)
        sched = ChronoScheduler(self.opt, self.cfg, total_steps=total_steps)
        self._tps_meter.reset()
        self._last_tok_per_sec = 0.0
        self._last_tok_per_sec_ema = 0.0

        pad_id = self.tok.PAD
        self.model.train()
        t0 = time.time()
        step = self._global_step
        for epoch in range(self.cfg.epochs if self.cfg.max_steps is None else 10**9):
            for bx, by in train_dl:
                p = curr_params(step + 1)
                if p["max_len"] < bx.size(1):
                    bx = bx[:, : p["max_len"]]
                    by = by[:, : p["max_len"]]

                self.psy.update(
                    token_dropout=self.cfg.token_dropout,
                    byte_noise=self.cfg.byte_noise,
                    span_mask_prob=self.cfg.span_mask_prob,
                    span_len=self.cfg.span_mask_len,
                )
                bx = self.psy.apply_input(bx, pad_id)

                B, T = bx.shape
                total_loss, total_tok = 0.0, 0
                accum = 0
                logits = None
                lvi_logs = {
                    "mv_reg": torch.tensor(0.0, device=self.device),
                    "two_view": torch.tensor(0.0, device=self.device),
                }
                oom_count = 0
                last_oom_err = None
                bx_cpu, by_cpu = bx, by

                while True:
                    total_loss = 0.0
                    total_tok = 0
                    accum = 0
                    idx = 0
                    restart_batch = False
                    batch_on_device = None

                    if self._prefetch_to_device:
                        try:
                            batch_on_device = (
                                bx_cpu.to(self.device, non_blocking=True),
                                by_cpu.to(self.device, non_blocking=True),
                            )
                        except RuntimeError as e:
                            if self._maybe_handle_oom(e, B * T):
                                restart_batch = True
                                oom_count += 1
                                last_oom_err = e
                                self.opt.zero_grad(set_to_none=True)
                                if self._scaler_enabled:
                                    self.scaler = GradScaler()
                                    print("[OOM] GradScaler reset after OOM")
                            else:
                                print(
                                    "[MPS] host batch prefetch disabled after error:",
                                    e,
                                )
                                self._prefetch_to_device = False
                            batch_on_device = None

                    if restart_batch:
                        if self._oom_retry_limit and oom_count > self._oom_retry_limit:
                            raise last_oom_err
                        continue

                    while idx < B:
                        current_micro = max(1, int(self._micro_active))
                        chunk_bs = max(1, math.ceil(B / current_micro))
                        end = min(B, idx + chunk_bs)
                        if batch_on_device is not None:
                            subx = batch_on_device[0][idx:end]
                            suby = batch_on_device[1][idx:end]
                        else:
                            subx = bx_cpu[idx:end].to(
                                self.device, non_blocking=True
                            )
                            suby = by_cpu[idx:end].to(
                                self.device, non_blocking=True
                            )

                        # AI pre-forward planning
                        try:
                            plan = self._ai.plan_pre_forward()
                            self._ai_fp32_logits = bool(plan.get("fp32_logits", False))
                        except Exception:
                            self._ai_fp32_logits = bool(
                                int(os.environ.get("AETHER_FP32_LOGITS", "0"))
                            )

                        try:
                            with torch.autocast(
                                device_type="mps", dtype=self.amp_dtype
                            ):
                                logits, lvi_logs = self._compute_logits(subx)
                                _pf = {}
                                try:
                                    _pf = self._ai.post_forward_assess(
                                        logits, subx=subx, suby=suby, pad_id=pad_id
                                    )
                                except Exception:
                                    pass
                            # ★ CE は常に FP32・autocast 無効で計算（数値安定のため）
                            with torch.autocast(device_type="mps", enabled=False):
                                loss_ce = self._ce_loss(logits, suby, pad_id)
                                loss = (
                                    loss_ce
                                    + float(self.cfg.lvi_mv_weight) * lvi_logs["mv_reg"]
                                    + float(self.cfg.lvi_two_view_weight)
                                    * lvi_logs["two_view"]
                                )

                                # ANI-AI observe forward
                                try:
                                    self._ai.observe_forward(loss, logits)
                                except Exception:
                                    pass

                                # --- K-bridge weak Huber regularizer (optional, safe default off)
                                if (
                                    self._k_enabled
                                    and _KBRIDGE_AVAILABLE
                                    and (self._k_reg_w > 0.0)
                                    and (self._teacher is not None)
                                ):
                                    try:
                                        with torch.no_grad():
                                            probs = torch.softmax(
                                                logits.float(), dim=-1
                                            )
                                            emb_w = self.model.emb.weight.float()
                                            E_pred = torch.einsum(
                                                "btv,vd->btd", probs, emb_w
                                            )
                                            E_bar = E_pred.mean(dim=1)
                                            t_pos = self._teacher(subx).to(
                                                E_bar.device, dtype=E_bar.dtype
                                            )
                                        loss = loss + float(self._k_reg_w) * khuber_loss(
                                            E_bar, t_pos, delta=1.0
                                        )
                                    except Exception:
                                        pass

                                if getattr(self.cfg, "intent_weight", 0.0) > 0.0:
                                    every = max(1, int(getattr(self.cfg, "intent_every", 1)))
                                    do_int = (every <= 1) or (
                                        self._global_step % every == 0
                                    )
                                    if do_int and (self._teacher is not None):
                                        try:
                                            emb_w = getattr(self.model, "emb", None)
                                            if emb_w is not None:
                                                t_pos = self._teacher(subx)
                                                neg_ids = _make_negative_ids(subx)
                                                t_neg = self._teacher(neg_ids)
                                                loss_int = _contrastive_intent_loss(
                                                    logits,
                                                    suby,
                                                    emb_w.weight,
                                                    pad_id,
                                                    t_pos,
                                                    t_neg,
                                                    margin=float(
                                                        getattr(
                                                            self.cfg,
                                                            "intent_margin",
                                                            0.10,
                                                        )
                                                    ),
                                                    sample_frac=float(
                                                        getattr(
                                                            self.cfg,
                                                            "intent_sample_frac",
                                                            0.25,
                                                        )
                                                    ),
                                                )
                                                loss = (
                                                    loss
                                                    + float(self.cfg.intent_weight)
                                                    * loss_int
                                                )
                                        except Exception:
                                            pass

                                loss = loss / max(1, current_micro)
                        except RuntimeError as e:
                            if self._maybe_handle_oom(e, B):
                                restart_batch = True
                                oom_count += 1
                                last_oom_err = e
                                self.opt.zero_grad(set_to_none=True)
                                if self._scaler_enabled:
                                    self.scaler = GradScaler()
                                    print("[OOM] GradScaler reset after OOM")
                                break
                            raise

                        if restart_batch:
                            break

                        if _pf.get("hazard", False):
                            try:
                                self._ai.sanitize_gradients(self.model)
                            except Exception:
                                pass
                            self.opt.zero_grad(set_to_none=True)
                            if self.scaler is not None:
                                self.scaler.update()
                        else:
                            if self.scaler is not None:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        # ANI-AI observe grads
                        try:
                            self._ai.observe_grads(self.model)
                        except Exception:
                            pass
                        self._zero_nonfinite_grads()
                        accum += 1

                        ntok = int((suby != pad_id).sum().item())
                        total_tok += ntok
                        total_loss += float(loss_ce.detach().cpu().item()) * ntok

                        current_micro = max(1, int(self._micro_active))
                        if accum >= current_micro:
                            if self.scaler is not None:
                                self.scaler.unscale_(self.opt)
                            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), self.cfg.grad_clip
                                )
                            # ANI-AI: skip step if requested
                            if hasattr(self, "_ai") and self._ai.should_skip():
                                if self.scaler is not None:
                                    self.scaler.update()
                                self.opt.zero_grad(set_to_none=True)
                            else:
                                if self.scaler is not None:
                                    self.scaler.step(self.opt)
                                    self.scaler.update()
                                else:
                                    self.opt.step()
                                self.opt.zero_grad(set_to_none=True)
                            # ANI-AI decide/act
                            try:
                                self._ai.decide_and_act(step)
                            except Exception:
                                pass
                            try:
                                if up is not None:
                                    up.mps_empty_cache_safe()
                            except Exception:
                                pass
                            if (
                                self._empty_cache_every > 0
                                and ((step + 1) % self._empty_cache_every == 0)
                                and self.device.type == "mps"
                            ):
                                try:
                                    torch.mps.empty_cache()
                                except Exception:
                                    pass
                            accum = 0

                        idx = end

                    if restart_batch:
                        if self._oom_retry_limit and oom_count > self._oom_retry_limit:
                            raise last_oom_err
                        continue
                    break

                if accum > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.opt)
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clip
                        )
                    # ANI-AI: skip step if requested
                    if hasattr(self, "_ai") and self._ai.should_skip():
                        if self.scaler is not None:
                            self.scaler.update()
                        self.opt.zero_grad(set_to_none=True)
                    else:
                        if self.scaler is not None:
                            self.scaler.step(self.opt)
                            self.scaler.update()
                        else:
                            self.opt.step()
                        self.opt.zero_grad(set_to_none=True)
                    # ANI-AI decide/act
                    try:
                        self._ai.decide_and_act(step)
                    except Exception:
                        pass
                    try:
                        if up is not None:
                            up.mps_empty_cache_safe()
                    except Exception:
                        pass
                    if (
                        self._empty_cache_every > 0
                        and ((step + 1) % self._empty_cache_every == 0)
                        and self.device.type == "mps"
                    ):
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass

                if self.cfg.mps_sync_every > 0 and (
                    step % self.cfg.mps_sync_every == 0
                ):
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass

                loss_avg = total_loss / max(1, total_tok)
                ppl = math.exp(min(20.0, loss_avg))
                elapsed = time.time() - t0
                t0 = time.time()
                tps = float(total_tok) / max(1e-6, elapsed)
                ema_tps = self._tps_meter.update(total_tok, elapsed)
                self._last_tok_per_sec = float(tps)
                self._last_tok_per_sec_ema = float(ema_tps)
                lr_mult = sched.step()
                step += 1
                self._global_step = step

                if self._adaptive_micro:
                    if self._micro_active > self._micro_base:
                        self._micro_stable += 1
                        if (
                            self._micro_recover_every > 0
                            and self._micro_stable >= self._micro_recover_every
                        ):
                            new_micro = max(self._micro_base, self._micro_active // 2)
                            if new_micro < self._micro_active:
                                self._micro_active = new_micro
                                print(
                                    f"[ADAPT] micro_batch relaxed to {self._micro_active}"
                                )
                            self._micro_stable = 0
                    else:
                        self._micro_stable = 0

                if int(getattr(self.cfg, "relora_every", 0)) > 0 and (
                    step % int(self.cfg.relora_every) == 0
                ):
                    try:
                        self._relora_cycle()
                    except Exception:
                        pass

                        # --- K metrics: token-level buffers（pmax/correct/pred/true/lenbin + groups）
                if self._k_enabled and _KBRIDGE_AVAILABLE:
                    try:
                        with torch.no_grad():
                            p = torch.softmax(logits.float(), dim=-1)
                            pmax = p.amax(dim=-1)  # (B,T)
                            pred = p.argmax(dim=-1)  # (B,T)
                            correct = (pred == suby).float()  # (B,T)
                            mask = suby != pad_id
                            # to CPU np
                            pm = (
                                pmax[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("float64", copy=False)
                            )
                            lb = (
                                correct[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("int32", copy=False)
                            )
                            pr = (
                                pred[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("int32", copy=False)
                            )
                            tr = (
                                suby[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("int32", copy=False)
                            )
                            # length-bin per token
                            import numpy as _np

                            lens = mask.sum(dim=1).detach().cpu().tolist()
                            edges = self._kb_len_edges
                            lenbins = []
                            for i in range(mask.size(0)):
                                L = int(lens[i])
                                if L <= 0:
                                    continue
                                b = 0
                                for j in range(len(edges) - 1):
                                    if edges[j] <= L < edges[j + 1]:
                                        b = j
                                        break
                                lenbins.append(
                                    _np.full(
                                        (int(mask[i].sum().item()),), b, dtype=_np.int32
                                    )
                                )
                            lbins = (
                                _np.concatenate(lenbins, axis=0)
                                if len(lenbins) > 0
                                else _np.zeros((0,), dtype=_np.int32)
                            )
                            # groups (map token id -> group id)
                            cm = self._kb_classmap
                            prg = cm[_np.clip(pr, 0, cm.size - 1)]
                            trg = cm[_np.clip(tr, 0, cm.size - 1)]
                            # append
                            self._kb_buf_pmax.append(pm)
                            self._kb_buf_labels.append(lb)
                            self._kb_buf_predcls.append(pr)
                            self._kb_buf_truecls.append(tr)
                            self._kb_buf_lenbin.append(lbins)
                            # store groups
                            self._kb_buf_predgrp = getattr(self, "_kb_buf_predgrp", [])
                            self._kb_buf_truegrp = getattr(self, "_kb_buf_truegrp", [])
                            self._kb_buf_predgrp.append(prg)
                            self._kb_buf_truegrp.append(trg)
                            # ring-buffer compress
                            total = sum(a.size for a in self._kb_buf_pmax)
                            if total > self._kb_buf_cap:
                                keep = self._kb_buf_cap // 2
                                cur = 0
                                start = 0
                                for i, a in enumerate(self._kb_buf_pmax):
                                    cur += a.size
                                    if cur >= (total - keep):
                                        start = i
                                        break
                                if start > 0:
                                    self._kb_buf_pmax = self._kb_buf_pmax[start:]
                                    self._kb_buf_labels = self._kb_buf_labels[start:]
                                    self._kb_buf_predcls = self._kb_buf_predcls[start:]
                                    self._kb_buf_truecls = self._kb_buf_truecls[start:]
                                    self._kb_buf_lenbin = self._kb_buf_lenbin[start:]
                                    self._kb_buf_predgrp = self._kb_buf_predgrp[start:]
                                    self._kb_buf_truegrp = self._kb_buf_truegrp[start:]
                    except Exception:
                        pass

                if step % self.cfg.log_every == 0:
                    # --- Temperature sweep ECE（current batch; snapshot only）
                    if self._k_enabled and _KBRIDGE_AVAILABLE and self._kb_tsweep:
                        try:
                            with torch.no_grad():
                                for Tval in self._kb_tsweep:
                                    pT = torch.softmax(
                                        (logits.float() / float(Tval)), dim=-1
                                    )
                                    pmaxT = pT.amax(dim=-1)
                                    predT = pT.argmax(dim=-1)
                                    correctT = (predT == suby).float()
                                    mask = suby != pad_id
                                    pmT = (
                                        pmaxT[mask]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .astype("float64", copy=False)
                                    )
                                    lbT = (
                                        correctT[mask]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .astype("int32", copy=False)
                                    )
                                    eT, _, _, _ = ece_and_hist_k(
                                        pmT, lbT, n_bins=max(2, self._kb_ece_bins)
                                  )
                                    try:
                                        V = int(logits.shape[-1])
                                        mask = (suby != pad_id)
                                        tmin = int(suby[mask].min().item()) if mask.any() else -1
                                        tmax = int(suby[mask].max().item()) if mask.any() else -1
                                        lmax = float(logits.detach().float().abs().amax().item())
                                        with torch.no_grad():
                                            p = torch.softmax(logits.detach().float(), dim=-1)
                                            pm = p[mask]
                                            ty = suby[mask]
                                            ce_manual = float(
                                                (
                                                    -torch.log(
                                                        pm[torch.arange(pm.size(0)), ty.view(-1)] + 1e-12
                                                    )
                                                )
                                                .mean()
                                                .item()
                                            )
                                            top1 = float(pm.max(dim=-1).values.mean().item())
                                        print(
                                            f"[DBG] V={V} target=[{tmin},{tmax}] | logits|max|={lmax:.2e} "
                                            f"| ce_manual={ce_manual:.3f} | top1={top1:.3f}"
                                        )
                                        assert tmax < V, f"target id {tmax} >= vocab {V}"
                                    except Exception as e:
                                        print("[DBG] sanity failed:", e)

                                    self.fabric.log(
                                        {f"eval/ece_T{Tval:g}": float(eT)}, step=step
                                    )
                        except Exception:
                            pass
                    # autosave LoRA safetensor
                    if int(getattr(self.cfg, "safetensor_every", 0)) > 0 and (
                        step % int(self.cfg.safetensor_every) == 0
                    ):
                        try:
                            save_lora_safetensors_if_any(
                                self.model,
                                getattr(self.cfg, "out_dir", "runs/v285"),
                                step,
                            )
                        except Exception as _e:
                            print("[SAFE] autosave error:", _e)

                    self.fabric.log(
                        {
                            "train/loss": float(loss_avg),
                            "train/ce": float(loss_avg),
                            "train/ppl": float(ppl),
                            "train/tok_s": float(tps),
                            "train/tok_s_ema": float(ema_tps),
                            "train/lr_mult": float(lr_mult),
                        },
                        step=step,
                    )

                    # --- K-bridge: ECE + hist（overall）/ length-bucket / classwise（拡張）
                    if (
                        self._k_enabled
                        and _KBRIDGE_AVAILABLE
                        and (step % max(1, self._kb_metrics_every) == 0)
                        and self._kb_buf_pmax
                    ):
                        try:
                            import numpy as _np

                            pm = _np.concatenate(self._kb_buf_pmax, axis=0)
                            lb = _np.concatenate(self._kb_buf_labels, axis=0)
                            pr = (
                                _np.concatenate(self._kb_buf_predcls, axis=0)
                                if getattr(self, "_kb_buf_predcls", None)
                                else _np.zeros((0,), dtype=_np.int32)
                            )
                            tr = (
                                _np.concatenate(self._kb_buf_truecls, axis=0)
                                if getattr(self, "_kb_buf_truecls", None)
                                else _np.zeros((0,), dtype=_np.int32)
                            )
                            ln = (
                                _np.concatenate(self._kb_buf_lenbin, axis=0)
                                if self._kb_buf_lenbin
                                else _np.zeros((0,), dtype=_np.int32)
                            )
                            prg = (
                                _np.concatenate(
                                    getattr(self, "_kb_buf_predgrp", []), axis=0
                                )
                                if getattr(self, "_kb_buf_predgrp", None)
                                else _np.zeros((0,), dtype=_np.int32)
                            )
                            trg = (
                                _np.concatenate(
                                    getattr(self, "_kb_buf_truegrp", []), axis=0
                                )
                                if getattr(self, "_kb_buf_truegrp", None)
                                else _np.zeros((0,), dtype=_np.int32)
                            )
                            # overall ECE + hist（等幅）
                            ece, counts, mconf, macc = ece_and_hist_k(
                                pm, lb, n_bins=max(2, self._kb_ece_bins)
                            )
                            logs = {"eval/ece_top": float(ece)}
                            for i, ct in enumerate(counts.tolist()[:10]):
                                logs[f"eval/conf_hist/bin{i}"] = float(ct)
                                logs[f"eval/conf_bin_mean/conf{i}"] = float(mconf[i])
                                logs[f"eval/conf_bin_mean/acc{i}"] = float(macc[i])
                            # quantile bins（分位ヒスト）
                            qspec = self._kb_qbins
                            if qspec:
                                if qspec.startswith("q"):
                                    try:
                                        Q = max(2, int(qspec[1:]))
                                    except Exception:
                                        Q = 10
                                    qs = _np.linspace(0.0, 1.0, Q + 1)
                                    edges = _np.quantile(pm, qs)
                                else:
                                    edges = _np.asarray(
                                        [
                                            float(x)
                                            for x in qspec.split(",")
                                            if x.strip()
                                        ],
                                        dtype=_np.float64,
                                    )
                                edges[0] = 0.0
                                edges[-1] = 1.0
                                e_q, cnt_q, mc_q, ma_q, edges = ece_and_hist_k_bins(
                                    pm, lb, edges
                                )
                                logs[f"eval/ece_quantile@{len(edges) - 1}"] = float(e_q)
                                for i, ct in enumerate(cnt_q.tolist()[:5]):
                                    logs[f"eval/qhist/bin{i}"] = float(ct)
                            # classwise（group or raw class）
                            base_key = self._kb_ece_by
                            if base_key in ("predgrp", "truegrp"):
                                g = prg if base_key == "predgrp" else trg
                                G = int(self._kb_group_count)
                                e_g, c_g = ece_multi_groups_k(
                                    pm,
                                    lb,
                                    g,
                                    n_groups=G,
                                    n_bins=max(2, self._kb_ece_bins),
                                )
                                if self._kb_ece_c_list:
                                    idxs = [
                                        i for i in self._kb_ece_c_list if 0 <= i < G
                                    ]
                                else:
                                    order = _np.argsort(-c_g)
                                    idxs = [
                                        int(i)
                                        for i in order[: max(1, self._kb_ece_c_top)]
                                        if c_g[int(i)] >= self._kb_ece_c_min
                                    ]
                                for gi in idxs[: max(1, self._kb_ece_c_top)]:
                                    name = (
                                        self._kb_group_names[gi]
                                        if gi < len(self._kb_group_names)
                                        else f"g{gi}"
                                    )
                                    logs[f"eval/ece_grp_{base_key}[{name}]"] = float(
                                        e_g[gi]
                                    )
                                    logs[f"eval/count_grp_{base_key}[{name}]"] = float(
                                        c_g[gi]
                                    )
                            # 2D: length × class-group
                            if (
                                self._kb_enable_2d
                                and len(ln) > 0
                                and base_key in ("predgrp", "truegrp")
                            ):
                                g = prg if base_key == "predgrp" else trg
                                edges = self._kb_len_edges
                                G = int(self._kb_group_count)
                                _, c_all = ece_multi_groups_k(
                                    pm,
                                    lb,
                                    g,
                                    n_groups=G,
                                    n_bins=max(2, self._kb_ece_bins),
                                )
                                if self._kb_ece_c_list:
                                    sel_groups = [
                                        i for i in self._kb_ece_c_list if 0 <= i < G
                                    ]
                                else:
                                    order = _np.argsort(-c_all)
                                    sel_groups = [
                                        int(i)
                                        for i in order[: max(1, self._kb_ece_c_top)]
                                        if c_all[int(i)] >= self._kb_ece_c_min
                                    ]
                                cells = 0
                                for b in range(len(edges) - 1):
                                    idx = ln == b
                                    if not idx.any():
                                        continue
                                    e_b, c_b = ece_multi_groups_k(
                                        pm[idx],
                                        lb[idx],
                                        g[idx],
                                        n_groups=G,
                                        n_bins=max(2, self._kb_ece_bins),
                                    )
                                    lo, hi = edges[b], edges[b + 1]
                                    cnt_len = float(c_b.sum())
                                    logs[f"eval/ece2d_len[{lo},{hi})/count"] = cnt_len
                                    for gi in sel_groups:
                                        name = (
                                            self._kb_group_names[gi]
                                            if gi < len(self._kb_group_names)
                                            else f"g{gi}"
                                        )
                                        logs[
                                            f"eval/ece2d_len[{lo},{hi})/grp[{name}]"
                                        ] = float(e_b[gi])
                                        cells += 1
                                        if cells >= self._kb_2d_max_cells:
                                            break
                                    if cells >= self._kb_2d_max_cells:
                                        break
                            # AUC/PR-AUC（overall）
                            try:
                                auc = roc_auc_binary(lb, pm)
                                pauc = pr_auc_binary(lb, pm)
                                logs["eval/auc"] = float(auc)
                                logs["eval/prauc"] = float(pauc)
                            except Exception:
                                pass
                            self.fabric.log(logs, step=step)
                        except Exception:
                            pass

                    # --- K-bridge: sequence-level NDCG@k (external hook provided)
                    if self._k_enabled and _KBRIDGE_AVAILABLE and self._kb_seq_scores:
                        try:
                            import numpy as _np

                            kseq = int(os.environ.get("KBRIDGE_NDCG_K", "10"))
                            Lmax = max(s.size for s in self._kb_seq_scores)
                            S = _np.stack(
                                [
                                    _np.pad(s, (0, Lmax - s.size), constant_values=0.0)
                                    for s in self._kb_seq_scores
                                ],
                                axis=0,
                            )
                            R = _np.stack(
                                [
                                    _np.pad(g, (0, Lmax - g.size), constant_values=0.0)
                                    for g in self._kb_seq_gains
                                ],
                                axis=0,
                            )
                            nd = ndcg_at_k_seq_k(S, R, k=kseq)
                            self.fabric.log(
                                {"eval/ndcg_seq@{}".format(kseq): float(nd.mean())},
                                step=step,
                            )
                            self._kb_seq_scores.clear()
                            self._kb_seq_gains.clear()
                        except Exception:
                            pass
                    if self.cfg.max_steps and step >= self.cfg.max_steps:
                        break
            if self.cfg.max_steps and step >= self.cfg.max_steps:
                break

    def _maybe_handle_oom(self, err: RuntimeError, batch_tokens: int) -> bool:
        if not getattr(self, "_adaptive_micro", False):
            return False
        msg = str(err).lower()
        keywords = (
            "out of memory",
            "mps backend out of memory",
            "failed to allocate",
            "resource exhausted",
        )
        if not any(k in msg for k in keywords):
            return False
        if self._micro_active >= self._micro_cap:
            print(
                f"[OOM] {err} -- micro_batch cap reached ({self._micro_active}); aborting"
            )
            return False
        prev = self._micro_active
        self._micro_active = min(
            self._micro_cap, max(prev + 1, prev * 2)
        )
        self._micro_stable = 0
        approx_chunk = max(1, math.ceil(batch_tokens / self._micro_active))
        print(
            f"[OOM] {err} -- increasing micro_batch to {self._micro_active} (chunk≈{approx_chunk})"
        )
        if self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return True

    def _zero_nonfinite_grads(self):
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                p.grad.zero_()

    @property
    def tok_per_sec(self) -> float:
        return float(self._last_tok_per_sec)

    @property
    def tok_per_sec_ema(self) -> float:
        return float(self._last_tok_per_sec_ema)


# ====== Lightning Fabric (optional placeholder) =============================
_HAVE_FABRIC = False
try:
    import lightning as L

    _HAVE_FABRIC = True
except Exception:
    L = None


class AetherTrainerFabric(AetherTrainerBase):
    def __init__(self, *a, ckpt_every: int = 1, precision: str = "16-mixed", **k):
        super().__init__(*a, **k)
        self.precision = precision

    def enable_lvi_for_trainer(self, cfg):
        pass


# ====== CLI ==================================================================


def save_lora_safetensors_if_any(model, out_dir: str, step: int):
    try:
        from safetensors.torch import save_file as _sf_save
    except Exception as _e:
        print("[SAFE] safetensors not available; skip save:", _e)
        return False
    try:
        import os

        os.makedirs(out_dir, exist_ok=True)
        sd = model.state_dict()
        sel = {
            k: v.detach().cpu()
            for k, v in sd.items()
            if ("lora_" in k) or ("adapter" in k)
        }
        if len(sel) == 0:
            print("[SAFE] no LoRA/adapter weights detected; skip")
            return False
        fn = os.path.join(out_dir, f"aether_lora_step{int(step)}.safetensors")
        _sf_save(sel, fn, metadata={"aether_ver": "2.8.5"})
        print(f"[SAFE] wrote LoRA safetensors: {fn}")
        return True
    except Exception as _e:
        print("[SAFE] save failed:", _e)
        return False


def __aether_main__():
    def _require_mps():
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise SystemExit("MPS is required. Run on macOS with Apple Silicon.")

    _require_mps()
    import argparse

    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--train_stream", action="store_true")
    ap.add_argument("--train_glob", type=str, default="")
    ap.add_argument("--val_glob", type=str, default="")
    ap.add_argument("--pack_len", type=int, default=1024)
    ap.add_argument("--buffer_size", type=int, default=8192)
    ap.add_argument("--loader_workers", type=int, default=0)
    ap.add_argument("--loader_prefetch", type=int, default=2)
    ap.add_argument("--loader_persistent_workers", action="store_true")
    ap.add_argument("--loader_pin_memory", action="store_true")
    ap.add_argument("--no_prefetch_to_device", action="store_true")
    # model
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--d_model", type=int, default=4096)
    ap.add_argument("--n_layers", type=int, default=32)
    ap.add_argument("--n_heads", type=int, default=32)
    ap.add_argument(
        "--kv_heads", type=int, default=0, help="0=disable GQA, >0=KV heads"
    )
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--use_rope", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10_000.0)
    ap.add_argument("--rope_scaling", type=float, default=1.0)
    ap.add_argument("--ff_mult", type=float, default=2.6666667)
    ap.add_argument("--use_abs_pos", action="store_true")
    # train
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--out_dir", type=str, default="runs/v28")
    ap.add_argument("--tiled_q", type=int, default=192)
    ap.add_argument("--tiled_k", type=int, default=320)
    # attention window
    ap.add_argument("--window_size", type=int, default=0)
    ap.add_argument("--global_tokens", type=int, default=0)
    ap.add_argument("--global_stride", type=int, default=0)
    ap.add_argument("--mps_sync_every", type=int, default=0)
    ap.add_argument("--allow_mps_cpu_fallback", action="store_true")
    # optimizer
    ap.add_argument("--opt_cpu8bit", action="store_true")
    ap.add_argument("--opt_galore", action="store_true")
    ap.add_argument("--galore_rank", type=int, default=64)
    # LVI
    ap.add_argument("--lvi", action="store_true")
    ap.add_argument("--lvi_k", type=int, default=64)
    ap.add_argument("--lvi_alpha_mode", type=str, default="sparsemax")
    ap.add_argument("--lvi_mv_weight", type=float, default=0.10)
    ap.add_argument("--lvi_two_view_weight", type=float, default=0.05)
    ap.add_argument("--lvi_every", type=int, default=4)
    # Intention
    ap.add_argument("--intent_weight", type=float, default=0.0)
    ap.add_argument("--intent_margin", type=float, default=0.10)
    ap.add_argument("--intent_sample_frac", type=float, default=0.25)
    ap.add_argument("--intent_every", type=int, default=4)
    # ReLoRA
    ap.add_argument("--relora_every", type=int, default=0)
    # LoRA/PEFT
    ap.add_argument("--peft_lora", action="store_true", help="Use PEFT-LoRA")
    ap.add_argument("--peft_targets", type=str, default="qkv,proj")
    ap.add_argument(
        "--int8_lora", action="store_true", help="Use custom INT8 base + LoRA"
    )
    ap.add_argument(
        "--hybrid_lora", action="store_true", help="PEFT + INT8-LoRA hybrid"
    )
    ap.add_argument("--lora_r", type=int, default=160)
    ap.add_argument("--lora_alpha", type=int, default=320)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_adapter", type=str, default="")
    # tracing
    ap.add_argument("--ts_trace", action="store_true")
    ap.add_argument("--ts_len", type=int, default=1024)
    ap.add_argument("--ts_out", type=str, default="runs/ts/aether_len{L}.pt")
    ap.add_argument("--ts_multi", type=str, default="")
    # memmap load
    ap.add_argument("--load_memmap", type=str, default="")
    ap.add_argument(
        "--safetensor_every",
        type=int,
        default=0,
        help="Save LoRA weights to safetensors every N steps (0=off)",
    )
    args = ap.parse_args()

    set_seed(1337)
    device = detect_device()

    tok = ByteTokenizer(vocab_size=args.vocab_size)
    kvh = args.kv_heads if args.kv_heads and args.kv_heads > 0 else None
    model = AetherPumpSimple(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=0.1,
        max_len=args.max_len,
        pad_id=tok.PAD,
        ff_mult=float(args.ff_mult),
        use_rope=bool(args.use_rope),
        rope_theta=float(args.rope_theta),
        rope_scaling=float(args.rope_scaling),
        use_abs_pos=bool(args.use_abs_pos),
        kv_heads=kvh,
    ).to(device)
    # ultramem autopatch from env (if enabled)
    if (up is not None) and (os.environ.get("ULTRAMEM_AUTOPATCH", "0") == "1"):
        up.install_autopatch_from_env(lambda: model)
        print("[ULTRAMEM] autopatch requested via env; model patched if configured")

    if args.load_memmap:
        sd = torch.load(args.load_memmap, map_location="cpu")
        model.load_state_dict(sd if isinstance(sd, dict) else sd["model"], strict=False)
        model.to(device)

    peft_targets = [s.strip() for s in args.peft_targets.split(",") if s.strip()]
    ultra_active = (up is not None) and (
        os.environ.get("ULTRAMEM_AUTOPATCH", "0") == "1"
    )
    if ultra_active:
        print("[ULTRAMEM] autopatch active; skipping built-in INT8/PEFT injections")
    else:
        if args.hybrid_lora:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not installed for --hybrid_lora")
            model = apply_hybrid_lora(
                model,
                peft_targets,
                int8_include=["w1", "w2", "w3"],
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=float(args.lora_dropout),
            )
        elif args.peft_lora and not args.int8_lora:
            model = apply_peft_lora(
                model,
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=float(args.lora_dropout),
                targets=peft_targets,
            )
        elif args.int8_lora and not args.peft_lora:
            n = convert_linear_to_int8_lora(
                model,
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=0.0,
                include_names=None,
                exclude_names=("emb", "head"),
                skip_if_out_equals=getattr(model, "vocab_size", None),
            )
            print(f"[INT8+LoRA] replaced Linear -> LinearInt8LoRA: {n}")
    model = model.to(device)
    # Freeze token embeddings (and tied output head)
    try:
        mod = model.base_model if hasattr(model, "base_model") else model
        for p in getattr(mod, "emb").parameters():
            p.requires_grad = False
        # head.weight は emb.weight と tie されてるので、これで出力側も自動で凍結される
        print("[FREEZE] embeddings frozen (LoRA-only training)")
    except Exception as e:
        print("[FREEZE] freeze failed:", e)
    cfg = TrainConfig(
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        lr=args.lr,
        warmup_steps=args.warmup,
        grad_clip=1.0,
        max_len=args.max_len,
        out_dir=args.out_dir,
        seed=1337,
        use_tb=False,
        tiled_q=args.tiled_q,
        tiled_k=args.tiled_k,
        mps_sync_every=int(args.mps_sync_every),
        lvi_mv_weight=args.lvi_mv_weight,
        lvi_two_view_weight=args.lvi_two_view_weight,
        window_size=int(args.window_size),
        global_tokens=int(args.global_tokens),
        global_stride=int(args.global_stride),
        opt_cpu8bit=bool(args.opt_cpu8bit),
        opt_galore=bool(args.opt_galore),
        galore_rank=int(args.galore_rank),
        intent_weight=float(args.intent_weight),
        intent_margin=float(args.intent_margin),
        intent_sample_frac=float(args.intent_sample_frac),
        intent_every=int(args.intent_every),
        relora_every=int(args.relora_every),
        kv_heads=kvh,
        lvi_enable=bool(args.lvi),
        lvi_k=int(args.lvi_k),
        lvi_alpha_mode=str(args.lvi_alpha_mode),
        lvi_every=int(args.lvi_every),
        loader_num_workers=int(args.loader_workers),
        loader_prefetch_factor=int(args.loader_prefetch),
        loader_persistent_workers=bool(args.loader_persistent_workers),
        loader_pin_memory=bool(args.loader_pin_memory),
        prefetch_to_device=not bool(args.no_prefetch_to_device),
        disallow_mps_fallback=not bool(args.allow_mps_cpu_fallback),
    )
    cfg.safetensor_every = int(getattr(args, "safetensor_every", 0))
    cfg.curriculum = [
        CurriculumStage(
            until_step=int(cfg.warmup_steps * 1.0),
            max_len=min(args.max_len, max(512, args.pack_len)),
        ),
        CurriculumStage(
            until_step=int(args.max_steps * 0.5) if args.max_steps else 10**9,
            max_len=min(args.max_len, max(1024, args.pack_len * 2)),
        ),
        CurriculumStage(until_step=args.max_steps or 10**9, max_len=args.max_len),
    ]
    cfg.peft_targets = args.peft_targets
    cfg.lora_r = int(args.lora_r)
    cfg.lora_alpha = int(args.lora_alpha)
    cfg.lora_dropout = float(args.lora_dropout)

    trainer = AetherTrainerMPS(model, tok, cfg, ckpt_every=1)

    # TorchScript trace
    if args.ts_trace:
        L = int(args.ts_len)
        x = torch.full((1, L), tok.PAD, dtype=torch.long, device=device)
        try:
            mt = torch.jit.trace(model, (x,), check_trace=False)
            os.makedirs(os.path.dirname(args.ts_out) or "runs/ts", exist_ok=True)
            torch.jit.save(mt, args.ts_out.replace("{L}", str(L)))
            print("[TS] saved:", args.ts_out)
        except Exception as e:
            print("[TS] trace failed:", e)
    if args.ts_multi:
        lens = [int(x.strip()) for x in args.ts_multi.split(",") if x.strip()]
        for L in lens:
            x = torch.full((1, L), tok.PAD, dtype=torch.long, device=device)
            try:
                mt = torch.jit.trace(model, (x,), check_trace=False)
                out_path = args.ts_out.replace("{L}", str(L))
                os.makedirs(os.path.dirname(out_path) or "runs/ts", exist_ok=True)
                torch.jit.save(mt, out_path)
                print("[TS] saved:", out_path)
            except Exception as e:
                print("[TS] trace failed:", e)

    # Train
    # --- Preflight: how many files will be used (見える化)
    n_tr = len(glob.glob(args.train_glob, recursive=True)) if args.train_glob else 0
    n_va = len(glob.glob(args.val_glob, recursive=True)) if args.val_glob else 0
    print(
        f"[DATA] train files={n_tr}  val files={n_va}  pattern_train={args.train_glob}  cwd={os.getcwd()}"
    )
    if args.train_stream and n_tr == 0:
        raise SystemExit(
            "[DATA] No training files found. Check --train_glob pattern or working directory."
        )
    if args.demo:
        s = "Hello, SpiralReality."
        ids = tok.encode(s)
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            y = model(x)
        print("demo logits:", y.shape)
    elif args.train_stream:
        assert args.train_glob, "--train_glob が必要です"
        ds_tr = StreamingTextDataset(
            args.train_glob,
            tok,
            pack_len=args.pack_len,
            buffer_size=args.buffer_size,
            infinite=True,
            seed=cfg.seed,
        )
        ds_va = (
            StreamingTextDataset(
                args.val_glob,
                tok,
                pack_len=min(4096, args.max_len),
                buffer_size=4096,
                infinite=False,
                seed=cfg.seed,
            )
            if args.val_glob
            else None
        )
        trainer.fit(ds_tr, ds_va)
        if args.save_adapter:
            save_peft_adapter_if_any(trainer.model, args.save_adapter)
    else:
        print(
            "Nothing to do. Use one of: --demo | --train_stream | --ts_trace | --ts_multi"
        )


# =============================================================================
if __name__ == "__main__":
    __aether_main__()
# =============================================================================


# ===================== Safety / ANI Extensions (Non-invasive hooks) =====================
# These utilities are appended without altering existing classes or logic.
# They are activated only if environment variables are set.

import os as _os
import json as _json
import types as _types
import time as _time


def _flatten_tensors(_x):
    if torch.is_tensor(_x):
        return [_x]
    elif isinstance(_x, (list, tuple)):
        out = []
        for z in _x:
            out.extend(_flatten_tensors(z))
        return out
    elif isinstance(_x, dict):
        out = []
        for z in _x.values():
            out.extend(_flatten_tensors(z))
        return out
    return []


def _safe_stats(t: torch.Tensor):
    try:
        if not torch.is_floating_point(t):
            return None
        finite = torch.isfinite(t)
        if finite.all():
            return None
        n = t.numel()
        nf = int((~finite).sum().item())
        ratio = nf / max(1, n)
        s = {"nf": nf, "n": n, "ratio": ratio}
        with torch.no_grad():
            s["min"] = float(torch.nan_to_num(t).min().item())
            s["max"] = float(torch.nan_to_num(t).max().item())
            s["mean"] = float(torch.nan_to_num(t).float().mean().item())
            s["std"] = float(
                torch.nan_to_num((t - t.float().mean()).float()).std().item()
            )
        return s
    except Exception:
        return {"nf": -1, "n": int(t.numel()) if t is not None else 0, "ratio": 1.0}


class _AetherEventBus:
    def __init__(self):
        self.handlers = []

    def emit(self, evt: dict):
        for h in list(self.handlers):
            try:
                h(evt)
            except Exception:
                pass

    def on(self, fn):
        self.handlers.append(fn)


_AETHER_EVENT_BUS = _AetherEventBus()


class _AetherNaNGuard:
    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "raise",
        check_inputs=True,
        check_outputs=True,
        check_grads=True,
        deep_patterns=None,
        deep_dump=False,
        deep_dump_on="event",
        outdir="runs/safety",
    ):
        self.model = model
        self.mode = mode
        self.check_inputs = bool(check_inputs)
        self.check_outputs = bool(check_outputs)
        self.check_grads = bool(check_grads)
        self.deep_patterns = [
            p.strip().lower() for p in (deep_patterns or []) if p.strip()
        ]
        self.deep_dump = bool(deep_dump)
        self.deep_dump_on = str(deep_dump_on or "event")
        self.outdir = outdir
        self._names = {id(m): n for n, m in model.named_modules()}
        self._pnames = {id(p): n for n, p in model.named_parameters()}
        self._installed = False
        _os.makedirs(_os.path.join(outdir, "tensors"), exist_ok=True)
        self._anomaly = False

    def _matches_deep(self, name: str) -> bool:
        if not self.deep_patterns:
            return False
        nl = name.lower()
        return any(p in nl for p in self.deep_patterns)

    def _nan2num_(self, t: torch.Tensor):
        if not torch.is_floating_point(t):
            return t
        torch.nan_to_num(
            t,
            nan=0.0,
            posinf=float(torch.finfo(t.dtype).max),
            neginf=float(torch.finfo(t.dtype).min),
            out=t,
        )
        return t

    def _clamp_(self, t: torch.Tensor, mn=None, mx=None):
        if not torch.is_floating_point(t):
            return t
        t.nan_to_num_(nan=0.0)
        if mn is None or mx is None:
            finfo = torch.finfo(t.dtype)
            mn = finfo.min if mn is None else mn
            mx = finfo.max if mx is None else mx
        t.clamp_(mn, mx)
        return t

    def _handle(self, where: str, name: str, tensors, stage=""):
        hazard = False
        stats = []
        for t in tensors:
            s = _safe_stats(t)
            if s is not None:
                hazard = True
                stats.append(s)
                if self.mode == "nan2num":
                    self._nan2num_(t)
                elif self.mode == "clamp":
                    self._clamp_(t)
        if hazard:
            evt = {
                "type": "nonfinite",
                "where": where,
                "name": name,
                "stage": stage,
                "stats": stats,
                "time": _time.time(),
            }
            _AETHER_EVENT_BUS.emit(evt)
            if (
                (self._matches_deep(name))
                and self.deep_dump
                and (self.deep_dump_on in ("always", "event"))
            ):
                # dump a small projection to save space
                try:
                    for i, t in enumerate(
                        [x for x in tensors if torch.is_tensor(x)][:2]
                    ):
                        path = _os.path.join(
                            self.outdir,
                            "tensors",
                            f"{int(evt['time'])}_{where}_{name.replace('.', '_')}_{stage}_{i}.pt",
                        )
                        torch.save(t.detach().cpu()[:2], path)
                        jpath = path.replace(".pt", ".json")
                        with open(jpath, "w") as f:
                            import json as __json

                            __json.dump(
                                {
                                    "name": name,
                                    "where": where,
                                    "stage": stage,
                                    "stats": stats,
                                },
                                f,
                            )
                except Exception:
                    pass
            if self.mode == "raise":
                raise FloatingPointError(
                    f"[NaNGuard] non-finite detected at {where}:{name} {stage} stats={stats[:1]}"
                )

    def _pre_hook(self, module, inputs):
        if not self.check_inputs:
            return
        name = self._names.get(id(module), module.__class__.__name__)
        tensors = _flatten_tensors(inputs)
        self._handle("forward_in", name, tensors, stage="pre")

    def _fwd_hook(self, module, inputs, outputs):
        if not self.check_outputs:
            return
        name = self._names.get(id(module), module.__class__.__name__)
        tensors = _flatten_tensors(outputs)
        # modify in-place for outputs if needed
        self._handle("forward_out", name, tensors, stage="post")

    def _grad_hook(self, p):
        def fn(g):
            if not self.check_grads:
                return g
            name = self._pnames.get(id(p), "param")
            s = _safe_stats(g)
            if s is not None:
                evt = {
                    "type": "nonfinite",
                    "where": "grad",
                    "name": name,
                    "stats": [s],
                    "time": _time.time(),
                }
                _AETHER_EVENT_BUS.emit(evt)
                if self.mode == "nan2num":
                    torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, out=g)
                elif self.mode == "clamp":
                    self._clamp_(g)
                elif self.mode == "raise":
                    raise FloatingPointError(
                        f"[NaNGuard] non-finite grad at {name} stats={s}"
                    )
            return g

        return fn

    def install(self, anomaly=False):
        if self._installed:
            return
        self._anomaly = bool(anomaly)
        if self._anomaly:
            try:
                torch.autograd.set_detect_anomaly(True)
            except Exception:
                pass
        for _, m in self.model.named_modules():
            try:
                m.register_forward_pre_hook(self._pre_hook, with_kwargs=False)
                m.register_forward_hook(self._fwd_hook, with_kwargs=False)
            except Exception:
                pass
        if self.check_grads:
            for p in self.model.parameters():
                if p.requires_grad:
                    try:
                        p.register_hook(self._grad_hook(p))
                    except Exception:
                        pass
        self._installed = True


class _AdaptiveBatchIter:
    """Wraps an IterableDataset to yield dynamic batch sizes driven by a callback."""

    def __init__(self, ds: IterableDataset, collate_fn, get_bs, shuffle: bool):
        self.ds = ds
        self.collate_fn = collate_fn
        self.get_bs = get_bs
        self.shuffle = shuffle
        self._it = None

    def __iter__(self):
        self._it = iter(self.ds)
        return self

    def __next__(self):
        bs = max(1, int(self.get_bs()))
        batch = []
        while len(batch) < bs:
            try:
                item = next(self._it)
            except StopIteration:
                # Recreate iterator if dataset is infinite; otherwise propagate
                if getattr(self.ds, "infinite", False):
                    self._it = iter(self.ds)
                    continue
                if len(batch) == 0:
                    raise
                else:
                    break
            batch.append(item)
        if len(batch) == 0:
            raise StopIteration
        return self.collate_fn(batch)

    def __len__(self):
        # Unknown for streaming; let caller handle
        raise TypeError


class _AetherANIManager:
    """Adaptive Nonfinite Intervention: dynamic batch shrink + loss scaling + event-driven escalation."""

    def __init__(self, trainer, enable: bool = True, outdir="runs/safety"):
        self.trainer = trainer
        self.model = trainer.model
        self.cfg = trainer.cfg
        self.outdir = outdir
        self.enabled = bool(enable)
        self.base_bs = int(getattr(self.cfg, "batch_size", 1))
        self.bs_div = 1  # effective_bs = base_bs // bs_div
        self.level = 0
        self.max_level = int(_os.getenv("AETHER_ANI_MAX_ESCALATION", "3"))
        self.patience = int(_os.getenv("AETHER_ANI_PATIENCE", "2"))
        self.cooldown = int(_os.getenv("AETHER_ANI_COOLDOWN", "200"))
        self.loss_scale = float(_os.getenv("AETHER_ANI_LOSS_SCALE", "1.0"))
        self.min_loss_scale = float(_os.getenv("AETHER_ANI_MIN_LOSS_SCALE", "0.0625"))
        self.scale_backoff = float(_os.getenv("AETHER_ANI_SCALE_BACKOFF", "0.5"))
        self.scale_growth = float(_os.getenv("AETHER_ANI_SCALE_GROWTH", "2.0"))
        self.last_hazard_step = -(10**9)
        self.step_counter = 0
        self.hazard_in_window = 0
        self.lr_backoff = float(
            _os.getenv("AETHER_ANI_LR_BACKOFF", "1.0")
        )  # no LR change by default
        self.skip_on_hazard = bool(int(_os.getenv("AETHER_ANI_SKIP_ON_HAZARD", "0")))
        self.grad_clip_val = float(_os.getenv("AETHER_ANI_GRAD_CLIP", "0.0"))
        self._install_log_path = _os.path.join(outdir, "ani_events.jsonl")
        _os.makedirs(outdir, exist_ok=True)
        self._install_logging()
        self._patch_clip_and_step()
        self._subscribe_events()

    # ---- logging ----
    def _install_logging(self):
        try:
            with open(self._install_log_path, "a") as f:
                f.write(
                    _json.dumps(
                        {
                            "t": _time.time(),
                            "event": "_install",
                            "base_bs": self.base_bs,
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    def log(self, d):
        try:
            with open(self._install_log_path, "a") as f:
                f.write(_json.dumps(d) + "\n")
        except Exception:
            pass

    # ---- batch size control ----
    def effective_bs(self) -> int:
        return max(1, self.base_bs // max(1, self.bs_div))

    def relax_if_stable(self):
        # called periodically by caller (e.g., each optimizer step)
        self.step_counter += 1
        since = self.step_counter - self.last_hazard_step
        if since >= max(self.cooldown, 1) and self.level > 0:
            self.level -= 1
            self.bs_div = max(1, self.bs_div // 2)
            self.loss_scale = min(1.0, self.loss_scale * self.scale_growth)
            self.log(
                {
                    "t": _time.time(),
                    "event": "relax",
                    "level": self.level,
                    "bs": self.effective_bs(),
                    "scale": self.loss_scale,
                }
            )

    # ---- hazard handling ----
    def _escalate(self, reason: str):
        if self.level < self.max_level:
            self.level += 1
            self.bs_div = min(2**self.level, max(1, self.base_bs))  # ceiling
            self.loss_scale = max(
                self.min_loss_scale, self.loss_scale * self.scale_backoff
            )
            # optional LR backoff
            if self.lr_backoff < 1.0:
                for g in self.trainer.opt.param_groups:
                    g["lr"] = g.get("_base_lr", g["lr"]) * self.lr_backoff
                    g["_base_lr"] = g["lr"]
            self.log(
                {
                    "t": _time.time(),
                    "event": "escalate",
                    "reason": reason,
                    "level": self.level,
                    "bs": self.effective_bs(),
                    "scale": self.loss_scale,
                }
            )

    def on_nonfinite(self, evt):
        self.last_hazard_step = self.step_counter
        self.hazard_in_window += 1
        self._escalate(f"{evt.get('where')}:{evt.get('name')}")
        if self.skip_on_hazard:
            # zero grads + skip this step by zeroing grads (training loop will still step, but grads are null)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    # ---- gradient scaling integration ----
    def unscale_grads(self, params_iter=None):
        if abs(self.loss_scale - 1.0) < 1e-6:
            return
        with torch.no_grad():
            ps = (
                list(self.model.parameters())
                if params_iter is None
                else list(params_iter)
            )
            for p in ps:
                if p.grad is not None:
                    p.grad.div_(self.loss_scale)

    def _patch_clip_and_step(self):
        # patch clip_grad_norm_ to unscale before clipping
        try:
            import torch.nn.utils as _nnutils

            _orig_clip = _nnutils.clip_grad_norm_

            def _patched_clip(params, max_norm, *a, **k):
                try:
                    if getattr(self.trainer, "_ani_manager", None) is self:
                        self.unscale_grads(params)
                except Exception:
                    pass
                return _orig_clip(params, max_norm, *a, **k)

            _nnutils.clip_grad_norm_ = _patched_clip
        except Exception:
            pass
        # patch optimizer.step to unscale grads always
        try:
            _orig_step = self.trainer.opt.step

            def _patched_step(*a, **k):
                try:
                    if getattr(self.trainer, "_ani_manager", None) is self:
                        self.unscale_grads()
                        self.relax_if_stable()
                except Exception:
                    pass
                return _orig_step(*a, **k)

            self.trainer.opt.step = _patched_step
        except Exception:
            pass

    # ---- dataloader patch ----
    def make_loader(self, ds, shuffle, max_len_for_collate):
        # If dataset is streaming (IterableDataset), return adaptive loader.
        is_iter = isinstance(ds, IterableDataset)
        if not is_iter:
            # fall back to original behavior
            return DataLoader(
                ds,
                batch_size=self.trainer.cfg.batch_size,
                shuffle=bool(shuffle),
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=lambda b: collate_lm_safe(b, pad_id=self.trainer.tok.PAD),
            )
        collate_fn = lambda b: collate_lm_safe(b, pad_id=self.trainer.tok.PAD)
        return _AdaptiveBatchIter(
            ds, collate_fn, get_bs=self.effective_bs, shuffle=bool(shuffle)
        )


def _install_nan_guard_and_ani_from_env(trainer):
    """Installer. Call once after trainer is constructed."""
    if int(_os.getenv("AETHER_NAN_GUARD", "0")):
        mode = _os.getenv("AETHER_NAN_GUARD_MODE", "raise")
        chk_in = bool(int(_os.getenv("AETHER_NAN_GUARD_CHECK_INPUTS", "1")))
        chk_out = bool(int(_os.getenv("AETHER_NAN_GUARD_CHECK_OUTPUTS", "1")))
        chk_grads = bool(int(_os.getenv("AETHER_NAN_GUARD_CHECK_GRADS", "1")))
        deep_en = bool(int(_os.getenv("AETHER_DEEP_ENABLE", "0")))
        deep_pats = (
            _os.getenv("AETHER_DEEP_PATTERNS", "attn,layernorm,loss").split(",")
            if deep_en
            else []
        )
        deep_dump = bool(int(_os.getenv("AETHER_DEEP_DUMP", "0")))
        deep_on = _os.getenv("AETHER_DEEP_DUMP_ON", "event")
        guard = _AetherNaNGuard(
            trainer.model,
            mode,
            chk_in,
            chk_out,
            chk_grads,
            deep_pats,
            deep_dump,
            deep_on,
        )
        guard.install(anomaly=bool(int(_os.getenv("AETHER_NAN_GUARD_ANOMALY", "0"))))
    if int(_os.getenv("AETHER_ANI_ENABLE", "0")):
        manager = _AetherANIManager(trainer, enable=True)
        trainer._ani_manager = manager
        # subscribe to NaN guard events
        _AETHER_EVENT_BUS.on(manager.on_nonfinite)
        # patch loader (instance-level)
        try:

            def _patched_make_loader(self, ds, shuffle, max_len_for_collate):
                return manager.make_loader(ds, shuffle, max_len_for_collate)

            trainer._make_loader = _types.MethodType(_patched_make_loader, trainer)
        except Exception:
            pass
        # global backward patch for loss scaling (scales loss before backward)
        try:
            _orig_backward = torch.Tensor.backward

            def _patched_backward(
                self, gradient=None, retain_graph=False, create_graph=False, inputs=None
            ):
                scale = (
                    getattr(trainer._ani_manager, "loss_scale", 1.0)
                    if getattr(trainer, "_ani_manager", None)
                    else 1.0
                )
                if isinstance(scale, (int, float)) and abs(scale - 1.0) > 1e-6:
                    self = self * float(scale)
                return _orig_backward(
                    self,
                    gradient=gradient,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                    inputs=inputs,
                )

            torch.Tensor.backward = _patched_backward
        except Exception:
            pass


# Inject installer call inside __aether_main__ dynamically by patching the function body at runtime is complex.
# Instead, we call the installer immediately after trainer creation by monkey-patching AetherTrainerMPS.__init__.
try:
    _orig_init = AetherTrainerMPS.__init__

    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        try:
            _install_nan_guard_and_ani_from_env(self)
        except Exception as _e:
            print("[ANI/NAN] installer failed:", _e)

    AetherTrainerMPS.__init__ = _patched_init
except Exception as _e:
    print("[ANI/NAN] __init__ patch failed:", _e)

# ===================== End of Safety / ANI Extensions =====================
# =============================================================================
# SpiralGuardian — Predictive NaN Guard + Smart ANI (loss/grad aware)
#  - pre-NaN detection (loss spike via EMA, sustained rise)
#  - grad collapse/explosion monitor
#  - staged escalation: lr_backoff / loss_scale backoff / grad_clip / step skip
#  - contextual death record + snapshots + optional restore
#  - non-invasive: opt.step() wrapped; trainer hooks optional
# =============================================================================
import collections
from typing import Dict, Optional
import torch
import torch.nn as nn


def _sg_now():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _sg_env_b(name: str, default: int = 0) -> bool:
    return str(os.environ.get(name, str(default))).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _sg_env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _sg_env_i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _sg_env_s(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _sg_cfg_from_env() -> Dict[str, Any]:
    droot = _sg_env_s("AETHER_SAFETY_DUMP_DIR", "runs/safety")
    os.makedirs(droot, exist_ok=True)
    os.makedirs(os.path.join(droot, "guardian"), exist_ok=True)
    os.makedirs(os.path.join(droot, "snaps"), exist_ok=True)
    return {
        "enable": _sg_env_b("AETHER_GUARDIAN", 1),
        # loss spike detector
        "ema_alpha": _sg_env_f("AETHER_GUARDIAN_EMA_ALPHA", 0.10),
        "spike_mult": _sg_env_f(
            "AETHER_GUARDIAN_LOSS_SPIKE_EMA_X", 1.75
        ),  # loss > ema*X
        "spike_delta": _sg_env_f(
            "AETHER_GUARDIAN_LOSS_SPIKE_DELTA", 0.20
        ),  # (loss-last)/max(last,eps) > delta
        "rise_window": _sg_env_i("AETHER_GUARDIAN_RISE_STEPS", 3),  # 連続上昇で発火
        # grad monitor
        "grad_min": _sg_env_f("AETHER_GUARDIAN_GRAD_MIN", 1e-7),
        "grad_max": _sg_env_f("AETHER_GUARDIAN_GRAD_MAX", 50.0),
        "grad_sample": _sg_env_i("AETHER_GUARDIAN_GRAD_SAMPLES", 128),  # 0=全件
        # staged actions
        "lr_backoff": _sg_env_f("AETHER_GUARDIAN_LR_BACKOFF", 0.5),
        "scale_backoff": _sg_env_f("AETHER_GUARDIAN_SCALE_BACKOFF", 0.5),
        "scale_min": _sg_env_f("AETHER_GUARDIAN_SCALE_MIN", 0.015625),
        "clip_on": _sg_env_b("AETHER_GUARDIAN_CLIP_ON", 1),
        "clip_value": _sg_env_f("AETHER_GUARDIAN_CLIP", 1.0),
        "skip_on": _sg_env_b("AETHER_GUARDIAN_SKIP", 1),
        "max_level": _sg_env_i("AETHER_GUARDIAN_MAX_LEVEL", 3),
        "patience": _sg_env_i("AETHER_GUARDIAN_PATIENCE", 1),  # 何件で次レベル
        "stop_after": _sg_env_i("AETHER_GUARDIAN_STOP_AFTER", 8),
        "cooldown": _sg_env_i("AETHER_GUARDIAN_COOLDOWN", 200),
        # snapshots
        "snap_every": _sg_env_i("AETHER_GUARDIAN_SNAP_EVERY", 500),
        "dump_dir": droot,
        "verbose": _sg_env_b("AETHER_GUARDIAN_VERBOSE", 1),
        # wire-up
        "wrap_opt": _sg_env_b("AETHER_GUARDIAN_WRAP_OPT", 1),
        "wrap_loss": _sg_env_b("AETHER_GUARDIAN_WRAP_LOSS", 1),
    }


class SpiralGuardian:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler=None,
        step_provider=None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.opt = optimizer
        self.sched = scheduler
        self.cfg = cfg or _sg_cfg_from_env()
        self.step_provider = step_provider
        self.dump_dir = os.path.join(self.cfg["dump_dir"], "guardian")
        os.makedirs(self.dump_dir, exist_ok=True)
        # state
        self.level = 0
        self.haz_count = 0
        self._stable = 0
        self._rise = 0
        self._ema = None
        self._last_loss = None
        self._skip_budget = 0
        self._snaps = collections.deque(maxlen=5)
        # patch optimizer
        if self.cfg["wrap_opt"] and self.opt is not None:
            self._wrap_optimizer()
        if self.cfg["verbose"]:
            print(
                "[GUARD] SpiralGuardian active | cfg:",
                {
                    k: self.cfg[k]
                    for k in (
                        "spike_mult",
                        "spike_delta",
                        "grad_min",
                        "grad_max",
                        "lr_backoff",
                        "scale_backoff",
                        "clip_value",
                        "patience",
                    )
                },
            )

    # ---- public hooks --------------------------------------------------------
    def observe_loss(self, loss: torch.Tensor):
        try:
            val = float(loss.detach().to("cpu"))
        except Exception:
            return
        st = self.step()
        # ema
        if self._ema is None:
            self._ema = val
        a = float(self.cfg["ema_alpha"])
        self._ema = (1.0 - a) * self._ema + a * val
        # spike heuristics
        spike_ema = val > self._ema * float(self.cfg["spike_mult"])
        spike_jump = False
        if self._last_loss is not None:
            denom = max(1e-9, abs(self._last_loss))
            spike_jump = ((val - self._last_loss) / denom) > float(
                self.cfg["spike_delta"]
            )
            self._rise = self._rise + 1 if val > self._last_loss else 0
        sustained = self._rise >= int(self.cfg["rise_window"])
        self._last_loss = val

        if spike_ema or spike_jump or sustained:
            self._report(
                "loss_spike",
                {
                    "step": st,
                    "loss": val,
                    "ema": self._ema,
                    "spike_ema": bool(spike_ema),
                    "spike_jump": bool(spike_jump),
                    "sustained": bool(sustained),
                },
            )

    def after_backward(self):
        # grad stats (sampled)
        gmin, gmax, gmean = None, None, 0.0
        count = 0
        limit = int(self.cfg["grad_sample"])
        for p in self.model.parameters():
            if p.grad is None or not p.grad.is_floating_point():
                continue
            gn = float(p.grad.detach().data.norm().cpu())
            gmin = gn if gmin is None else min(gmin, gn)
            gmax = gn if gmax is None else max(gmax, gn)
            gmean += gn
            count += 1
            if limit > 0 and count >= limit:
                break
        if count > 0:
            gmean /= count
            if not math.isfinite(gmean) or (
                gmax is not None and not math.isfinite(gmax)
            ):
                self._report("grad_nonfinite", {"gmean": gmean, "gmax": gmax})
            else:
                if gmean < float(self.cfg["grad_min"]):
                    self._report("grad_collapse", {"gmean": gmean})
                if gmax is not None and gmax > float(self.cfg["grad_max"]):
                    self._report("grad_explosion", {"gmax": gmax})

    # ---- core ---------------------------------------------------------------
    def step(self) -> int:
        try:
            if self.step_provider:
                return int(self.step_provider())
        except Exception:
            pass
        return int(getattr(self, "_fallback_step", 0))

    def _report(self, kind: str, payload: Dict[str, Any]):
        self.haz_count += 1
        if self.cfg["verbose"]:
            print(f"[GUARD] hazard {kind} @step={self.step()} :: {payload}")
        # write context
        self._write_context(kind, payload)
        # escalate w/ patience
        if (self.haz_count % max(1, int(self.cfg["patience"]))) == 0:
            self._escalate(kind)

        # stop criteria
        if self.haz_count >= int(self.cfg["stop_after"]):
            fn = self._snapshot(reason="stop_after")
            print("[GUARD] stop-after reached; snapshot:", fn)
            raise SystemExit(2)

    def _escalate(self, reason: str):
        self.level = min(int(self.cfg["max_level"]), self.level + 1)
        if self.cfg["verbose"]:
            print(f"[GUARD] ESCALATE → L{self.level} ({reason})")
        # staged actions
        self._apply_lr_backoff()
        self._apply_scale_backoff()
        if self.cfg["clip_on"]:
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg["clip_value"])
                )
            except Exception:
                pass
        if self.cfg["skip_on"]:
            self._skip_budget = max(
                self._skip_budget, 1
            )  # skip next step at least once
        # snapshot on every escalation
        self._snapshot(reason=f"escalate_L{self.level}")

    def _forgive(self):
        # cooldown forgiveness: decay level, restore scales partly
        if self.level > 0:
            self.level -= 1
        if self.cfg["verbose"]:
            print(f"[GUARD] FORGIVE → L{self.level}")
        # allow loss_scale growth next steps; lr gradually returns via sched

    # ---- optimizer wrapping --------------------------------------------------
    def _wrap_optimizer(self):
        if getattr(self.opt, "_sg_wrapped", False):
            return
        orig_step = self.opt.step
        guardian = self

        def wrapped_step(*args, **kwargs):
            # pre
            if guardian._skip_budget > 0:
                if guardian.cfg["verbose"]:
                    print("[GUARD] skip optimizer.step() due to hazard")
                guardian._skip_budget -= 1
                guardian.opt.zero_grad(set_to_none=True)
                return
            # pre-check grads
            bad = False
            for p in guardian.model.parameters():
                g = p.grad
                if g is None or not g.is_floating_point():
                    continue
                if not torch.isfinite(g).all():
                    bad = True
                    break
            if bad:
                guardian._report("grad_nonfinite_pre_step", {})
                guardian.opt.zero_grad(set_to_none=True)
                return
            # step
            res = orig_step(*args, **kwargs)
            # post: sanity of params
            badp = False
            for p in guardian.model.parameters():
                if not p.is_floating_point():
                    continue
                if not torch.isfinite(p).all():
                    badp = True
                    break
            if badp:
                guardian._report("param_nonfinite_post_step", {})
                # restore last snap if any
                if guardian._snaps:
                    guardian._restore(guardian._snaps[-1])
                    guardian.opt.zero_grad(set_to_none=True)
            # periodic snap + cooldown
            guardian._cooldown_tick()
            guardian._periodic_snap()
            return res

        self.opt.step = wrapped_step
        self.opt._sg_wrapped = True
        if self.cfg["verbose"]:
            print("[GUARD] optimizer wrapped")

    def _apply_lr_backoff(self):
        try:
            for pg in self.opt.param_groups:
                base = pg.get("_base_lr", pg.get("lr", 1e-3))
                if "_base_lr" not in pg:
                    pg["_base_lr"] = base
                factor = float(self.cfg["lr_backoff"]) ** max(1, self.level)
                pg["lr"] = float(base) * factor
        except Exception:
            pass

    def _apply_scale_backoff(self):
        # AMP loss-scaler互換（trainer側にscaleがあれば使う）
        try:
            scaler = getattr(self, "scaler", None)
            if scaler is None and hasattr(self.opt, "loss_scaler"):
                scaler = getattr(self.opt, "loss_scaler", None)
            if scaler is None:
                # try global in trainer (AETHER_ANI_LOSS_SCALE 系と連携)
                scaler = getattr(self, "_trainer", None)
                if scaler is not None:
                    scaler = getattr(scaler, "loss_scaler", None)
            if scaler is None:
                return
            cur = float(getattr(scaler, "scale", getattr(scaler, "_scale", 1.0)))
            new = max(
                float(self.cfg["scale_min"]), cur * float(self.cfg["scale_backoff"])
            )
            try:
                setattr(scaler, "scale", new)
            except Exception:
                try:
                    scaler._scale = torch.tensor(new, device="cpu")
                except Exception:
                    pass
            if self.cfg["verbose"]:
                print(f"[GUARD] loss_scale backoff {cur} → {new}")
        except Exception:
            pass

    # ---- snapshots & logs ----------------------------------------------------
    def _snapshot(self, reason="manual"):
        try:
            st = self.step()
            path = os.path.join(
                self.cfg["dump_dir"],
                "snaps",
                f"{_sg_now()}_step{st}_L{self.level}_{reason}.pt",
            )
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict()
                    if self.opt is not None
                    else None,
                    "meta": {"step": st, "level": self.level, "reason": reason},
                },
                path,
            )
            self._snaps.append(path)
            while len(self._snaps) > 5:
                self._snaps.popleft()
            if self.cfg["verbose"]:
                print("[GUARD] snapshot:", path)
            return path
        except Exception as e:
            print("[GUARD] snapshot failed:", e)

    def _restore(self, path: str):
        try:
            data = torch.load(path, map_location="cpu")
            self.model.load_state_dict(data.get("model", {}))
            if self.opt is not None and data.get("optimizer") is not None:
                try:
                    self.opt.load_state_dict(data["optimizer"])
                except Exception as e:
                    print("[GUARD] opt restore failed:", e)
            if self.cfg["verbose"]:
                print("[GUARD] restored:", path)
        except Exception as e:
            print("[GUARD] restore failed:", e)

    def _write_context(self, kind, payload):
        try:
            st = self.step()
            meta = {
                "t": _sg_now(),
                "step": st,
                "kind": kind,
                "payload": payload,
                "level": self.level,
                "lr_groups": [
                    pg.get("lr", None) for pg in getattr(self.opt, "param_groups", [])
                ],
                "clip": float(self.cfg["clip_value"]),
                "scale_min": float(self.cfg["scale_min"]),
            }
            with open(
                os.path.join(self.dump_dir, f"{_sg_now()}_step{st}_{kind}.json"), "w"
            ) as f:
                json.dump(meta, f)
        except Exception:
            pass

    def _cooldown_tick(self):
        self._stable += 1
        if self._stable >= int(self.cfg["cooldown"]):
            self._stable = 0
            # forgive one level
            self._forgive()

    def _periodic_snap(self):
        st = self.step()
        se = int(self.cfg["snap_every"])
        if se > 0 and (st % se) == 0:
            self._snapshot(reason="periodic")


# ---- installer ---------------------------------------------------------------
def _install_spiral_guardian_from_env(trainer):
    cfg = _sg_cfg_from_env()
    if not cfg["enable"]:
        return None
    try:
        model = getattr(trainer, "model", None)
        opt = getattr(trainer, "opt", None)
        sched = getattr(trainer, "sched", None) if hasattr(trainer, "sched") else None
        guard = SpiralGuardian(
            model,
            opt,
            sched,
            step_provider=lambda: int(getattr(trainer, "_global_step", 0)),
            cfg=cfg,
        )
        # wire loss observation if requested
        if cfg["wrap_loss"]:
            try:
                _orig = trainer._ce_loss

                def _wrap(self, logits, targets, pad_id):
                    out = _orig(logits, targets, pad_id)
                    guard.observe_loss(out)
                    return out

                import types as _types

                trainer._ce_loss = _types.MethodType(_wrap, trainer)
                print("[GUARD] loss wrapper installed")
            except Exception as e:
                print("[GUARD] loss wrapper failed:", e)
        # wire backward hook
        try:
            _orig_bw = (
                trainer.fit_one_batch if hasattr(trainer, "fit_one_batch") else None
            )
        except Exception:
            _orig_bw = None
        # we can call after_backward() from trainer loop; if not, offer a public handle
        setattr(trainer, "guardian_after_backward", lambda: guard.after_backward())
        # stash for external control
        setattr(trainer, "_spiral_guardian", guard)
        print("[GUARD] SpiralGuardian installed")
        return guard
    except Exception as e:
        print("[GUARD] install failed:", e)
        return None


# =============================================================================
