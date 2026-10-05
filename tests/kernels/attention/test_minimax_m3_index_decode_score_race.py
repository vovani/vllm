# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexDecodeScoreKernel under concurrent write-heavy work on another stream.

The consumer warps release a shared-memory stage (empty mbarrier arrive) right
after issuing their ldmatrix reads.  Without a generic->async proxy fence the
producer's next TMA into that stage can land while the reads are still in
flight, so the reads return parts of the next page.  The window is widened by
write-heavy kernels running concurrently (here: in-place ``mul_`` over 256 MB on
a side stream); without the fence about half of the launches return wrong
scores at these shapes on B200.
"""

import pytest
import torch

from vllm.platforms import current_platform

BLOCK_SIZE = 128


def _reference(q, cache, block_table, seq_lens, dql, ncols):
    """score[h, t, j] = max over the causally visible keys of page j of q.k (fp64)."""
    total_q, heads, _ = q.shape
    out = torch.full(
        (heads, total_q, ncols), float("-inf"), dtype=torch.float64, device=q.device
    )
    qf = q.to(torch.float64)
    for b in range(block_table.shape[0]):
        seq_len = int(seq_lens[b])
        nblk = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        k = cache[block_table[b, :nblk].long()].to(torch.float64)
        logits = torch.einsum("ihd,jrd->hijr", qf[b * dql : (b + 1) * dql], k)
        kpos = torch.arange(nblk * BLOCK_SIZE, device=q.device).view(nblk, BLOCK_SIZE)
        qpos = seq_len - dql + torch.arange(dql, device=q.device)
        allowed = kpos.view(1, nblk, BLOCK_SIZE) <= qpos.view(dql, 1, 1)
        logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        out[:, b * dql : (b + 1) * dql, :nblk] = logits.amax(-1)
    return out


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="CuteDSL index decode score requires Blackwell (sm100).",
)
@pytest.mark.parametrize(
    ("num_heads", "dql"),
    [(1, 4), (4, 1)],  # M3 TP4 + MTP (production), TP1 without MTP
)
def test_index_decode_score_under_concurrent_writes(num_heads: int, dql: int):
    pytest.importorskip("cutlass")
    from vllm.models.minimax_m3.nvidia.ops import (
        minimax_m3_index_decode_score_cutedsl,
    )

    torch.manual_seed(0)
    dev = "cuda"
    batch, blocks_per_req, ncols = 4, 1040, 8192
    total_q = batch * dql
    num_pages = batch * blocks_per_req + 1  # page 0 unused (the null block)
    q = (torch.randn(total_q, num_heads, 128, device=dev) * 0.5).to(torch.float8_e4m3fn)
    cache = (torch.randn(num_pages, BLOCK_SIZE, 128, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(num_pages - 1, device=dev).to(torch.int32) + 1
    block_table = perm[: batch * blocks_per_req].view(batch, blocks_per_req)
    block_table = block_table.contiguous()
    max_tokens = (blocks_per_req - 14) * BLOCK_SIZE
    seq_lens = torch.tensor(
        [max_tokens - 2 - 37 * i for i in range(batch)], dtype=torch.int32, device=dev
    )
    # Production layout: token-major [T, H, MK] buffer, the kernel writes the
    # transposed [H, T, MK] view.
    unified = torch.empty(total_q, num_heads, ncols, device=dev)

    def run():
        unified.fill_(float("-inf"))
        minimax_m3_index_decode_score_cutedsl(
            q,
            cache,
            block_table,
            seq_lens,
            max_tokens,
            0,
            1,
            num_heads,
            dql,
            dql,
            score_out=unified.transpose(0, 1),
        )

    expected = (
        _reference(q, cache, block_table, seq_lens, dql, ncols)
        .to(torch.float32)
        .transpose(0, 1)
        .contiguous()
    )
    run()
    torch.cuda.synchronize()
    quiet = unified.clone()
    torch.testing.assert_close(quiet, expected, atol=2e-3, rtol=0)

    noise_buf = torch.zeros(1 << 26, device=dev)  # 256 MB fp32
    side = torch.cuda.Stream()
    iters, per_iter = 60, 5
    outs = torch.empty(per_iter, *unified.shape, device=dev)
    wrong = 0
    for _ in range(iters):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(40):
                noise_buf.mul_(1.0)  # in-place read+write, runs concurrently
        for k in range(per_iter):
            run()
            outs[k].copy_(unified)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        wrong += sum(int(not torch.equal(outs[k], quiet)) for k in range(per_iter))
    assert wrong == 0, (
        f"{wrong}/{iters * per_iter} launches under concurrent writes differ from "
        "the quiet launch"
    )
