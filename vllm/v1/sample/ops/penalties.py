# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.model_executor.layers.utils import apply_penalties

_cached_buf: np.ndarray | None = None
_cached_lens: np.ndarray | None = None
_cached_gpu_tensor: torch.Tensor | None = None
_cached_gpu_shape: tuple[int, int] = (0, 0)


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors_incremental(
        output_token_ids, vocab_size, logits.device
    )

    # Async scheduling may leave -1 placeholders; replace with vocab_size
    # so the scatter in apply_penalties stays in-bounds.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors_incremental(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    Incrementally maintain a GPU tensor across steps. Only copies the
    new token(s) per request each step via a small CPU→GPU transfer,
    instead of rebuilding and copying the entire (batch, seq_len) matrix.
    """
    global _cached_buf, _cached_lens, _cached_gpu_tensor, _cached_gpu_shape

    n_reqs = len(output_token_ids)
    if n_reqs == 0:
        return torch.zeros((0, 0), dtype=torch.int64, device=device)

    max_len = max(len(ids) for ids in output_token_ids)

    # --- Ensure CPU buffer is large enough ---
    if (
        _cached_buf is None
        or _cached_buf.shape[0] < n_reqs
        or _cached_buf.shape[1] < max_len
    ):
        new_rows = max(n_reqs, _cached_buf.shape[0] if _cached_buf is not None else 0)
        new_cols = max(max_len, _cached_buf.shape[1] if _cached_buf is not None else 0)
        new_cols = ((new_cols + 1023) // 1024) * 1024

        new_buf = np.full((new_rows, new_cols), vocab_size, dtype=np.int64)
        new_lens = np.zeros(new_rows, dtype=np.int64)

        if _cached_buf is not None:
            copy_rows = min(_cached_buf.shape[0], new_rows)
            copy_cols = min(_cached_buf.shape[1], new_cols)
            new_buf[:copy_rows, :copy_cols] = _cached_buf[:copy_rows, :copy_cols]
            new_lens[:copy_rows] = _cached_lens[:copy_rows]

        _cached_buf = new_buf
        _cached_lens = new_lens

    # --- Ensure GPU tensor is large enough ---
    need_gpu_realloc = (
        _cached_gpu_tensor is None
        or _cached_gpu_shape[0] < n_reqs
        or _cached_gpu_shape[1] < max_len
    )
    if need_gpu_realloc:
        gpu_rows = max(n_reqs, _cached_gpu_shape[0])
        gpu_cols = max(max_len, _cached_gpu_shape[1])
        gpu_cols = ((gpu_cols + 1023) // 1024) * 1024
        new_gpu = torch.full(
            (gpu_rows, gpu_cols), vocab_size, dtype=torch.int64, device=device
        )
        if _cached_gpu_tensor is not None:
            cr = min(_cached_gpu_shape[0], gpu_rows)
            cc = min(_cached_gpu_shape[1], gpu_cols)
            new_gpu[:cr, :cc] = _cached_gpu_tensor[:cr, :cc]
        _cached_gpu_tensor = new_gpu
        _cached_gpu_shape = (gpu_rows, gpu_cols)

    # --- Incremental update: only copy new tokens ---
    updates_row = []
    updates_col = []
    updates_val = []

    for i in range(n_reqs):
        ids = output_token_ids[i]
        cur_len = len(ids)
        prev_len = int(_cached_lens[i])

        if cur_len < prev_len:
            # Request was replaced — mark for full rewrite.
            _cached_buf[i, :cur_len] = ids
            _cached_buf[i, cur_len:prev_len] = vocab_size
            for j in range(cur_len):
                updates_row.append(i)
                updates_col.append(j)
                updates_val.append(ids[j])
            for j in range(cur_len, prev_len):
                updates_row.append(i)
                updates_col.append(j)
                updates_val.append(vocab_size)
        elif cur_len > prev_len:
            for j in range(prev_len, cur_len):
                _cached_buf[i, j] = ids[j]
                updates_row.append(i)
                updates_col.append(j)
                updates_val.append(ids[j])

        _cached_lens[i] = cur_len

    if _cached_buf.shape[0] > n_reqs:
        _cached_lens[n_reqs:] = 0

    # --- Apply updates to GPU tensor ---
    if updates_val:
        rows_t = torch.tensor(updates_row, dtype=torch.int64, device=device)
        cols_t = torch.tensor(updates_col, dtype=torch.int64, device=device)
        vals_t = torch.tensor(updates_val, dtype=torch.int64, device=device)
        _cached_gpu_tensor[rows_t, cols_t] = vals_t

    return _cached_gpu_tensor[:n_reqs, :max_len]


