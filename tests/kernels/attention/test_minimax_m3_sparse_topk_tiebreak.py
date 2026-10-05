# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fmha_sm100 sparse_topk_select: exact score ties at the top-k boundary.

The MiniMax-M3 indexer selects 16 KV blocks per token with
``sparse_topk_select`` (IndexerTopKWithSortKernel).  When several blocks have
bitwise-equal scores at the 16th place, the kernel must resolve the tie by a
fixed rule (higher score first, then the lower block index), not by
shared-memory atomic arrival order: identical inputs must always select the
same blocks.  Each case below plants such a tie and drives one of the kernel's
selection paths (all-pairs rank loop, warp merge, refinement steps 1-3).
"""

import pytest
import torch

from vllm.platforms import current_platform

TOPK = 16
FLT_MAX = torch.finfo(torch.float32).max

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="fmha_sm100 sparse_topk_select requires Blackwell (sm100).",
)


def _reference(scores, nvp, force_begin, force_end):
    """Top-16 per row: forced blocks, then higher score, then lower index;
    returned ascending, -1 padded (the kernel's output contract)."""
    s = scores.cpu()
    T, H, K = s.shape
    out = torch.full((T, H, TOPK), -1, dtype=torch.int32)
    for t in range(T):
        n = min(int(nvp[t]), K)
        for h in range(H):
            if n <= TOPK:
                out[t, h, :n] = torch.arange(n, dtype=torch.int32)
                continue
            idx = torch.arange(n)
            fes = n - force_end if force_end <= n else 0
            forced = (idx < force_begin) | (idx >= fes)
            eff = torch.where(forced, torch.tensor(FLT_MAX), s[t, h, :n])
            # stable descending sort: equal scores keep ascending index order
            order = torch.sort(eff, descending=True, stable=True).indices
            out[t, h] = order[:TOPK].sort().values.to(torch.int32)
    return out


def _boundary_tied(row, n, force_begin, force_end):
    idx = torch.arange(n)
    forced = (idx < force_begin) | (idx >= n - force_end)
    eff = torch.where(forced, torch.tensor(FLT_MAX), row[:n])
    v = torch.sort(eff, descending=True).values
    return bool(v[TOPK - 1] == v[TOPK])


def _rows(kind: str, T: int, K: int, n: int, g: torch.Generator) -> torch.Tensor:
    """T rows of width K (n valid, -inf tail) with an exact tie at the boundary."""
    s = torch.full((T, 1, K), float("-inf"))
    for t in range(T):
        r = torch.rand(n, generator=g)
        hot = torch.randperm(n, generator=g)
        if kind == "rank_loop":  # spread scores: small threshold bin
            r = torch.randn(n, generator=g) * 4
        elif kind == "warp_merge":  # 1500 in one fp16 bin: 417..2048 staged
            r[hot[:1500]] = 36 + 2 * torch.rand(1500, generator=g)
        elif kind == "step2_rank_loop":  # overflow steps 0-1, small step-2 bins
            r[hot[:3000]] = 36 + 0.5 * torch.rand(3000, generator=g)
        elif kind == "step2_warp_merge":  # 1000 in one step-2 bin
            r[hot[:1000]] = 37.9921875 + 0.0078125 * torch.rand(1000, generator=g)
            r[hot[1000:3500]] = 36 + 1.9 * torch.rand(2500, generator=g)
        elif kind == "step3_exact":  # 3000 bitwise-equal scores
            r[hot[:3000]] = 36.075134
        elif kind == "step3_low_bits":  # equal in all but the 2 lowest bits
            base = torch.tensor([36.075134]).view(torch.int32) & ~3
            low = torch.randint(0, 4, (3000,), generator=g, dtype=torch.int32)
            r[hot[:3000]] = (base + low).view(torch.float32)
        else:
            raise ValueError(kind)
        if not kind.startswith("step3"):
            # copy the 16th-ranked score (the last block is forced) to 4 more blocks
            body = r[: n - 1]
            rank = torch.sort(body, descending=True, stable=True).indices
            v = body[rank[TOPK - 2]].clone()
            r[torch.randperm(n - 1, generator=g)[:4]] = v
        s[t, 0, :n] = r
        assert _boundary_tied(s[t, 0], n, 0, 1), kind
    return s.cuda().contiguous()


@pytest.mark.parametrize(
    ("kind", "K", "n"),
    [
        ("rank_loop", 8192, 1027),  # M3 decode: 1027 of 8192 blocks valid
        ("warp_merge", 4096, 4096),
        ("step2_rank_loop", 4096, 4096),
        ("step2_warp_merge", 4096, 4096),
        ("step3_exact", 4096, 4096),
        ("step3_low_bits", 4096, 4096),
    ],
)
def test_sparse_topk_select_boundary_ties_are_deterministic(kind, K, n):
    from vllm.third_party.fmha_sm100.api import sparse_topk_select

    g = torch.Generator().manual_seed(0)
    T, reps = 16, 1000
    scores = _rows(kind, T, K, n, g)
    nvp = torch.full((T,), n, dtype=torch.int32, device="cuda")
    outs = torch.empty(reps, T, 1, TOPK, dtype=torch.int32, device="cuda")
    for i in range(reps):
        sparse_topk_select(
            scores,
            TOPK,
            num_valid_pages=nvp,
            force_begin_blocks=0,
            force_end_blocks=1,
            output=outs[i],
            max_score_layout="THK",
        )
    torch.cuda.synchronize()
    differing = int((outs != outs[0]).flatten(1).any(1).sum())
    assert differing == 0, f"{differing}/{reps} runs differ from the first"
    expected = _reference(scores, nvp, 0, 1)
    assert torch.equal(outs[0].cpu(), expected), "tie not resolved by lowest index"


@pytest.mark.parametrize("seed", range(4))
def test_sparse_topk_select_matches_reference_without_ties(seed):
    from vllm.third_party.fmha_sm100.api import sparse_topk_select

    g = torch.Generator().manual_seed(seed)
    for T, K in ((16, 8192), (64, 1040), (8, 4096), (3, 33)):
        scores = torch.full((T, 1, K), float("-inf"))
        scores[:, :, : K - 7] = torch.randn(T, 1, K - 7, generator=g) * 10
        scores = scores.cuda().contiguous()
        nvp = torch.randint(1, K - 6, (T,), generator=g, dtype=torch.int32).cuda()
        expected = _reference(scores, nvp, 1, 1)
        assert not any(
            _boundary_tied(scores[t, 0].cpu(), int(nvp[t]), 1, 1)
            for t in range(T)
            if int(nvp[t]) > TOPK
        )
        out = sparse_topk_select(
            scores,
            TOPK,
            num_valid_pages=nvp,
            force_begin_blocks=1,
            force_end_blocks=1,
            max_score_layout="THK",
        )
        assert torch.equal(out.cpu(), expected)
