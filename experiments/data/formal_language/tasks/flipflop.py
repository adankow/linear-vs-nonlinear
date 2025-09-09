# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Pöppel, Andreas Auer
from typing import Optional

import numpy as np


# State transition of a Flip-Flop monoid, initially at low.
# V = \{high, low, id\}
# encoding:
# 0 -> PAD      2 -> high
# 1 -> low      3 -> id

def flipflop(
    *,
    batch_size: int = 1,
    vocab_size: int = 4,
    min_sequence_length: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    context_length: int = 20,
    pad_idx: int = 0,
    seed: int = 42,
    **kwargs
):
    vocab = np.arange(0, vocab_size - 1)
    rng = np.random.default_rng(seed)

    max_sequence_length = context_length if max_sequence_length is None else max_sequence_length
    min_sequence_length = max_sequence_length if min_sequence_length is None else min_sequence_length

    res = np.zeros([batch_size, context_length], dtype=np.int32)
    res[:, :-1] = rng.integers(1, vocab_size, size=[batch_size, context_length - 1])
    sizes = rng.integers(min_sequence_length, max_sequence_length + 1, size=[batch_size])
    prediction_mask = np.zeros_like(res)
    for batch_idx in range(batch_size):
        res[batch_idx, sizes[batch_idx] - 1 :] = np.zeros_like(res[batch_idx, sizes[batch_idx] - 1 :])
        trunc_res = res[batch_idx, : sizes[batch_idx] - 1]
        state = trunc_res[ trunc_res != 3]
        res[batch_idx, sizes[batch_idx] - 1] = 1 if state.shape[0] == 0 else state[-1]
        prediction_mask[batch_idx, sizes[batch_idx] - 1 : sizes[batch_idx]] = 1
    return res, prediction_mask
