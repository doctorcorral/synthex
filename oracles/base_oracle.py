#!/usr/bin/env python3
"""Shared oracle infrastructure for Synthex environment adapters.

Provides predicate/feature evaluation, chain-action dispatch,
parallelized batch scoring, and the standard JSON-over-tempfile
entrypoint used by all oracle scripts.
"""

import sys
import json
from multiprocessing import Pool, cpu_count


# ── Feature and predicate evaluation ─────────────────────────────

def eval_feature(feat, state):
    kind = feat[0]
    if kind == "axis":
        return state[feat[1]] < feat[2]
    elif kind == "diag":
        return feat[3] * state[feat[1]] + state[feat[2]] < 0
    elif kind == "sq_diag":
        return feat[3] * state[feat[1]] ** 2 + state[feat[2]] < 0
    elif kind == "prod":
        return state[feat[1]] * state[feat[2]] < feat[3]
    elif kind == "tridiag":
        return feat[4] * state[feat[1]] + feat[5] * state[feat[2]] + state[feat[3]] < 0
    elif kind == "swap_outcome":
        return eval_pred(feat[2], state)
    elif kind == "swap_outcome_neg":
        return not eval_pred(feat[2], state)
    elif kind == "sin_axis":
        import math
        return math.sin(state[feat[1]]) < feat[2]
    elif kind == "cos_axis":
        import math
        return math.cos(state[feat[1]]) < feat[2]
    elif kind == "wavelet_box":
        # Haar/box indicator: lo <= obs[i] < hi
        return feat[2] <= state[feat[1]] < feat[3]
    elif kind == "wavelet_ricker":
        # Ricker/Mexican-hat: psi((obs[i]-b)/a) < t,
        # psi(z) = (1 - z**2) * exp(-z**2 / 2)
        import math
        z = (state[feat[1]] - feat[2]) / feat[3]
        return (1.0 - z * z) * math.exp(-(z * z) / 2.0) < feat[4]
    return False


def eval_pred(pred, state):
    if pred is None or pred == "truep":
        return True
    if pred == "falsep":
        return False
    kind = pred[0]
    if kind == "feat":
        return eval_feature(pred[1], state)
    if kind == "not":
        return not eval_pred(pred[1], state)
    if kind == "and":
        return eval_pred(pred[1], state) and eval_pred(pred[2], state)
    if kind == "or":
        return eval_pred(pred[1], state) or eval_pred(pred[2], state)
    return False


def chain_action(chain, default, obs):
    for pred, action in chain:
        if eval_pred(pred, obs):
            return action
    return default


# ── Parallelized batch scoring ───────────────────────────────────

def score_batch_parallel(worker_fn, batch_args, sequential_threshold=4):
    """Run worker_fn over batch_args, using multiprocessing when beneficial.

    Falls back to sequential execution when len(batch_args) <= sequential_threshold.
    """
    n = len(batch_args)
    if n == 0:
        return []
    if n <= sequential_threshold:
        return [worker_fn(a) for a in batch_args]
    n_workers = min(cpu_count(), n, 8)
    with Pool(processes=n_workers) as pool:
        return pool.map(worker_fn, batch_args,
                        chunksize=max(1, n // (n_workers * 4)))


# ── Standard JSON-over-tempfile entrypoint ───────────────────────

def run_oracle(dispatch_fn):
    """Read request JSON from sys.argv[1], dispatch, write response to sys.argv[2]."""
    with open(sys.argv[1]) as f:
        request = json.load(f)

    result = dispatch_fn(request)

    with open(sys.argv[2], 'w') as f:
        json.dump(result, f)
