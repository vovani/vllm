# Tiled dstate=96 variant of _selective_scan_update_kernel.
# Tiles dstate=96 as 3x32 (zero waste). All tiles unrolled, loaded upfront.
# Kept in a separate file to avoid affecting Triton's compilation
# of the original kernel in mamba_ssm.py.

from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


@triton.jit
def fast_exp(x):
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens_ptr"] is not None}
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.jit(do_not_specialize=["N"])
def _selective_scan_update_kernel_tiled(
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
    z_ptr, out_ptr, state_batch_indices_ptr, dst_state_batch_indices_ptr,
    pad_slot_id, num_accepted_tokens_ptr, cu_seqlens_ptr,
    N, nheads, dim, dstate, nheads_ngroups_ratio,
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    stride_state_indices_batch, stride_state_indices_T,
    stride_dst_state_indices_batch, stride_dst_state_indices_T,
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """dstate=96 hardcoded: 3 unrolled tiles of 32 (zero waste).
    All state tiles loaded upfront for contiguous HBM reads.
    Outer seq_len loop, inner unrolled tiles: x/dt loaded once per token.
    """
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = eos - bos
        if seq_len == 0:
            return
    else:
        bos = pid_b
        seq_len = 1

    state_ptr_base = state_ptr

    if HAS_STATE_BATCH_INDICES:
        if IS_SPEC_DECODING:
            num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
            init_token_idx = tl.maximum(num_accepted - 1, 0)
        else:
            init_token_idx = 0
        dst_state_batch_indices_ptr += pid_b * stride_dst_state_indices_batch
        if not IS_SPEC_DECODING:
            dst_state_batch_idx = tl.load(
                dst_state_batch_indices_ptr
                + init_token_idx * stride_dst_state_indices_T
            ).to(tl.int64)
            dst_state_ptr = state_ptr + (
                dst_state_batch_idx * stride_state_batch + pid_h * stride_state_head)
        state_batch_indices_ptr += (
            pid_b * stride_state_indices_batch + init_token_idx * stride_state_indices_T)
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        dst_state_ptr = state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < dim

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim

    if not IS_SPEC_DECODING:
        dst_state_ptrs_base = dst_state_ptr
    if HAS_D:
        D = tl.load(D_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    offs_n0 = tl.arange(0, 32)
    offs_n1 = 32 + tl.arange(0, 32)
    offs_n2 = 64 + tl.arange(0, 32)

    state_mask = m_mask[:, None]
    if HAS_STATE_BATCH_INDICES:
        state_mask = state_mask & (state_batch_idx != pad_slot_id)

    state0 = tl.load(state_ptr + offs_m[:, None] * stride_state_dim + offs_n0[None, :] * stride_state_dstate, mask=state_mask, other=0.0).to(tl.float32)
    state1 = tl.load(state_ptr + offs_m[:, None] * stride_state_dim + offs_n1[None, :] * stride_state_dstate, mask=state_mask, other=0.0).to(tl.float32)
    state2 = tl.load(state_ptr + offs_m[:, None] * stride_state_dim + offs_n2[None, :] * stride_state_dstate, mask=state_mask, other=0.0).to(tl.float32)

    out_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for i_t in range(seq_len):
        x = tl.load(x_ptr + offs_m * stride_x_dim, mask=m_mask, other=0.0).to(tl.float32)

        B0 = tl.load(B_ptr + offs_n0 * stride_B_dstate).to(tl.float32)
        B1 = tl.load(B_ptr + offs_n1 * stride_B_dstate).to(tl.float32)
        B2 = tl.load(B_ptr + offs_n2 * stride_B_dstate).to(tl.float32)

        if not TIE_HDIM:
            dt_val = tl.load(dt_ptr + offs_m * stride_dt_dim, mask=m_mask, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt_val += tl.load(dt_bias_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt_val = softplus(dt_val)

            A0 = tl.load(A_ptr + offs_m[:, None] * stride_A_dim + offs_n0[None, :] * stride_A_dstate, mask=m_mask[:, None], other=0.0).to(tl.float32)
            A1 = tl.load(A_ptr + offs_m[:, None] * stride_A_dim + offs_n1[None, :] * stride_A_dstate, mask=m_mask[:, None], other=0.0).to(tl.float32)
            A2 = tl.load(A_ptr + offs_m[:, None] * stride_A_dim + offs_n2[None, :] * stride_A_dstate, mask=m_mask[:, None], other=0.0).to(tl.float32)

            dt_2d = dt_val[:, None]
            dB0 = B0[None, :] * dt_2d
            dB1 = B1[None, :] * dt_2d
            dB2 = B2[None, :] * dt_2d
            state0 = state0 * fast_exp(A0 * dt_2d) + dB0 * x[:, None]
            state1 = state1 * fast_exp(A1 * dt_2d) + dB1 * x[:, None]
            state2 = state2 * fast_exp(A2 * dt_2d) + dB2 * x[:, None]
        else:
            dt_val = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt_val += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt_val = softplus(dt_val)
            dA = fast_exp(tl.load(A_ptr).to(tl.float32) * dt_val)
            dB0 = B0 * dt_val
            dB1 = B1 * dt_val
            dB2 = B2 * dt_val

            state0 = state0 * dA + dB0 * x[:, None]
            state1 = state1 * dA + dB1 * x[:, None]
            state2 = state2 * dA + dB2 * x[:, None]

        if i_t == seq_len - 1:
            C0 = tl.load(C_ptr + offs_n0 * stride_C_dstate).to(tl.float32)
            C1 = tl.load(C_ptr + offs_n1 * stride_C_dstate).to(tl.float32)
            C2 = tl.load(C_ptr + offs_n2 * stride_C_dstate).to(tl.float32)
            out_acc = (tl.sum(state0 * C0[None, :], axis=1)
                     + tl.sum(state1 * C1[None, :], axis=1)
                     + tl.sum(state2 * C2[None, :], axis=1))

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch

    if not IS_SPEC_DECODING:
        dst0 = dst_state_ptrs_base + offs_m[:, None] * stride_state_dim + offs_n0[None, :] * stride_state_dstate
        dst1 = dst_state_ptrs_base + offs_m[:, None] * stride_state_dim + offs_n1[None, :] * stride_state_dstate
        dst2 = dst_state_ptrs_base + offs_m[:, None] * stride_state_dim + offs_n2[None, :] * stride_state_dstate
        tl.store(dst0, state0.to(dst0.dtype.element_ty), mask=state_mask)
        tl.store(dst1, state1.to(dst1.dtype.element_ty), mask=state_mask)
        tl.store(dst2, state2.to(dst2.dtype.element_ty), mask=state_mask)

    out_ptrs = out_ptr + (seq_len - 1) * stride_out_batch + offs_m * stride_out_dim
    x_last = tl.load(
        x_ptr - stride_x_batch + offs_m * stride_x_dim,
        mask=m_mask, other=0.0).to(tl.float32)
    out = out_acc
    if HAS_D:
        out += x_last * D
    if HAS_Z:
        z_ptrs = z_ptr + (seq_len - 1) * stride_z_batch + offs_m * stride_z_dim
        z = tl.load(z_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=m_mask)
