# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# sage_quant_mxfp4_fp8_input: same contract as sage_quant_mxfp4 but accepts
# fp8 q and k tensors. The fp8->fp32 widening happens inside the Triton
# kernels on the first tl.load, so no intermediate bf16 tensor is written to
# HBM.
#
# v stays bf16 — it is quantized to fp8 by sage_quant_v_kernel as usual.
# k-smoothing (subtract k.mean) cannot be done in fp8 on the host, so the
# mean is computed in fp32 and subtracted inside the kernel (smooth_k=True
# case defers the subtraction to the Triton kernel via a pre-computed k_mean).

import torch
import triton
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import map_dims
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    sage_quant_v_kernel,
)
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant_fp8_input import (
    _rotate_quantize_q_fp8_kernel,
    _rotate_quantize_k_fp8_kernel,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import create_hadamard_matrix


def sage_quant_mxfp4_fp8_input(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    R=None,
    BLOCK_R=128,
):
    """
    Quantize fp8 q/k and bf16 v for mxfp4 sage attention.

    q, k must be fp8 (float8_e4m3fn or float8_e4m3fnuz).
    v must be bf16 or fp16.

    Returns: q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s
      - q_fp4, k_fp4 : uint8 packed e2m1 fp4
      - q_scale, k_scale : uint8 e8m0 block scales  [*, D/32]
      - v_fp8  : fp8 quantized v
      - v_scale: fp32 per-(B,H,D) scale
      - delta_s: None (q_smoothing not yet supported for fp8 input path)
    """
    assert q.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ), f"q must be fp8, got {q.dtype}"
    assert k.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ), f"k must be fp8, got {k.dtype}"
    assert not q_smoothing, "q_smoothing is not supported for fp8 input"

    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    b, s_q, h_q, d = map_dims(q.shape, bshd_map)
    _, s_k, h_k, _ = map_dims(k.shape, bshd_map)

    if sm_scale is None:
        sm_scale = d ** -0.5

    if R is None:
        R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=torch.bfloat16) / (
            BLOCK_R ** 0.5
        )

    # --- k smoothing: compute mean in fp32 (fp8 arithmetic unsupported on host)
    #     and subtract before passing to the kernel.  We store the result as fp8
    #     so the kernel still loads fp8 from HBM.
    k_mean = k.to(torch.float32).mean(
        dim=1 if layout == "bshd" else 2, keepdim=True
    )
    k_smoothed = (k.to(torch.float32) - k_mean).to(k.dtype)

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k_smoothed.stride(), bshd_map)

    Q_NUM_BLKS = (s_q + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (s_k + BLKK - 1) // BLKK

    # Allocate outputs
    Q_q = q.new_empty((*q.shape[:-1], d // 2), dtype=torch.uint8)
    Q_descale = q.new_empty((*q.shape[:-1], d // BLOCK_R), dtype=torch.uint8)
    K_q = k.new_empty((*k.shape[:-1], d // 2), dtype=torch.uint8)
    K_descale = k.new_empty((*k.shape[:-1], d // BLOCK_R), dtype=torch.uint8)

    stride_qqb, stride_qqm, stride_qqh, stride_qqd = map_dims(Q_q.stride(), bshd_map)
    stride_kqb, stride_kqn, stride_kqh, stride_kqd = map_dims(K_q.stride(), bshd_map)
    stride_qsb, stride_qsm, stride_qsh, stride_qsd = map_dims(Q_descale.stride(), bshd_map)
    stride_ksb, stride_ksn, stride_ksh, stride_ksd = map_dims(K_descale.stride(), bshd_map)

    # Q kernel
    grid_q = (b * h_q * Q_NUM_BLKS,)
    _rotate_quantize_q_fp8_kernel[grid_q](
        q,
        Q_q,
        Q_descale,
        None,   # Q_mean — q_smoothing disabled
        R,
        sm_scale * 1.4426950408889634,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_qqb, stride_qqm, stride_qqh, stride_qqd,
        stride_qsb, stride_qsm, stride_qsh, stride_qsd,
        0, 0, 0, 0,   # mean strides (unused)
        b, h_q, s_q, d,
        q_smoothing=False,
        BLOCK_M=BLKQ,
        BLOCK_R=BLOCK_R,
        D=d,
        num_warps=4,
        num_stages=5,
    )

    # K kernel (receives already-smoothed k_smoothed in fp8)
    grid_k = (b * h_k * K_NUM_BLKS,)
    _rotate_quantize_k_fp8_kernel[grid_k](
        k_smoothed,
        K_q,
        K_descale,
        R,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_kqb, stride_kqn, stride_kqh, stride_kqd,
        stride_ksb, stride_ksn, stride_ksh, stride_ksd,
        b, h_k, s_k, d,
        smooth_k=False,   # already subtracted above
        BLOCK_M=BLKK,
        BLOCK_R=BLOCK_R,
        D=d,
        num_warps=4,
        num_stages=5,
    )

    # V quantization (unchanged — v is already bf16)
    stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = map_dims(v.stride(), bshd_map)
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE)
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_k * K_NUM_BLKS
    sage_quant_v_kernel[(v_task_count,)](
        v,
        v_fp8,
        v_scale,
        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v,
        v_scale.stride(0), v_scale.stride(1),
        b, h_k,
        K_NUM_BLKS,
        s_k,
        D=d,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    return Q_q, Q_descale, K_q, K_descale, v_fp8, v_scale, None
