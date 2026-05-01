#!/usr/bin/env python3
"""Generic Gymnasium adapter for MuJoCo continuous-action environments
with binary-weighted action decomposition.

Supports any MuJoCo env via the 'env_name' field in the JSON request.

Environment-specific config:
  InvertedPendulum-v5:  4D state,  1D action in [-3, +3]
  Swimmer-v5:           8D state,  2D action in [-1, +1]
  Hopper-v5:           11D state,  3D action in [-1, +1]
  HalfCheetah-v5:      17D state,  6D action in [-1, +1]
  Walker2d-v5:         17D state,  6D action in [-1, +1]

Commands: collect_states, score_bit, info
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import eval_pred, run_oracle, score_batch_parallel

import numpy as np
import gymnasium as gym

ENV_CONFIGS = {
    "InvertedPendulum-v5": {
        "num_dims": 4,
        "n_action_dims": 1,
        "action_low": -3.0,
        "action_high": 3.0,
        "max_steps": 1000,
        "success_threshold": 950,
    },
    "Swimmer-v5": {
        "num_dims": 8,
        "n_action_dims": 2,
        "action_low": -1.0,
        "action_high": 1.0,
        "max_steps": 1000,
        "success_threshold": 50,
    },
    "Hopper-v5": {
        "num_dims": 11,
        "n_action_dims": 3,
        "action_low": -1.0,
        "action_high": 1.0,
        "max_steps": 1000,
        "success_threshold": 500,
        "env_kwargs": {"healthy_reward": 0.0},
    },
    "HalfCheetah-v5": {
        "num_dims": 17,
        "n_action_dims": 6,
        "action_low": -1.0,
        "action_high": 1.0,
        "max_steps": 1000,
        "success_threshold": 1000,
    },
    "Walker2d-v5": {
        "num_dims": 17,
        "n_action_dims": 6,
        "action_low": -1.0,
        "action_high": 1.0,
        "max_steps": 1000,
        "success_threshold": 500,
    },
}


def bits_to_action(bit_values, cfg, bits_per_dim):
    weights = [2**i for i in range(bits_per_dim)]
    max_sum = sum(weights)
    n_action_dims = cfg["n_action_dims"]
    lo, hi = cfg["action_low"], cfg["action_high"]

    actions = np.zeros(n_action_dims)
    for d in range(n_action_dims):
        s = 0
        for i in range(bits_per_dim):
            s += weights[i] * bit_values[d * bits_per_dim + i]
        actions[d] = lo + (hi - lo) * s / max_sum
    return actions


def bit_policy_action(bit_preds, obs, cfg, bits_per_dim):
    bits = [1 if eval_pred(p, obs) else 0 for p in bit_preds]
    return bits_to_action(bits, cfg, bits_per_dim)


def _make_env(env_name):
    cfg = ENV_CONFIGS[env_name]
    return gym.make(env_name, **cfg.get("env_kwargs", {}))


def _run_episode_bits(args):
    env_name, bit_preds, seed, max_steps, bits_per_dim = args
    cfg = ENV_CONFIGS[env_name]
    env = _make_env(env_name)
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    for _ in range(max_steps):
        action = bit_policy_action(bit_preds, obs.tolist(), cfg, bits_per_dim)
        obs, r, term, trunc, _ = env.step(action)
        total_r += r
        if term or trunc:
            break
    env.close()
    return total_r


def run_episodes_bits(env_name, bit_preds, seeds, max_steps, bits_per_dim):
    cfg = ENV_CONFIGS[env_name]
    total = 0.0
    successes = 0
    for s in seeds:
        r = _run_episode_bits((env_name, bit_preds, s, max_steps, bits_per_dim))
        total += r
        if r > cfg["success_threshold"]:
            successes += 1
    return total, successes


def collect_states_bits(env_name, bit_preds, seeds, max_steps, bits_per_dim):
    cfg = ENV_CONFIGS[env_name]
    all_states = []
    n_success = 0
    for seed in seeds:
        env = _make_env(env_name)
        obs, _ = env.reset(seed=seed)
        ep_r = 0.0
        for _ in range(max_steps):
            all_states.append(obs.tolist())
            action = bit_policy_action(bit_preds, obs.tolist(), cfg, bits_per_dim)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc:
                break
        env.close()
        if ep_r > cfg["success_threshold"]:
            n_success += 1
    return all_states, n_success


def _score_bit_one(args):
    idx, candidate, env_name, bit_preds, target_bit, seeds, max_steps, bits_per_dim = args
    test_preds = list(bit_preds)
    test_preds[target_bit] = candidate
    reward, successes = run_episodes_bits(env_name, test_preds, seeds, max_steps, bits_per_dim)
    return {"idx": idx, "reward": reward, "landings": successes}


def score_bit_batch(env_name, candidates, bit_preds, target_bit, seeds,
                    max_steps, bits_per_dim):
    args_list = [
        (i, cand, env_name, bit_preds, target_bit, seeds, max_steps, bits_per_dim)
        for i, cand in enumerate(candidates)
    ]
    return score_batch_parallel(_score_bit_one, args_list)


def dispatch(request):
    cmd = request["cmd"]
    env_name = request.get("env_name", "InvertedPendulum-v5")
    bits_per_dim = request.get("bits_per_dim", 3)
    cfg = ENV_CONFIGS[env_name]
    n_bits = bits_per_dim * cfg["n_action_dims"]
    max_steps = request.get("max_steps", cfg["max_steps"])

    if cmd == "collect_states":
        bit_preds = request.get("bit_predicates")
        if bit_preds is None:
            bit_preds = ["falsep"] * n_bits
        seeds = request.get("seeds", list(range(40)))
        states, n_success = collect_states_bits(
            env_name, bit_preds, seeds, max_steps, bits_per_dim)
        return {"states": states, "n_landings": n_success,
                "n_episodes": len(seeds)}

    elif cmd == "score_bit":
        candidates = request["candidates"]
        bit_preds = request["bit_predicates"]
        target_bit = request["target_bit"]
        seeds = request.get("seeds", list(range(30)))
        baseline_reward, baseline_success = run_episodes_bits(
            env_name, bit_preds, seeds, max_steps, bits_per_dim)
        scores = score_bit_batch(
            env_name, candidates, bit_preds, target_bit, seeds,
            max_steps, bits_per_dim)
        return {"scores": scores, "baseline_reward": baseline_reward,
                "baseline_landings": baseline_success}

    elif cmd == "info":
        return {
            "env_name": env_name,
            "num_dims": cfg["num_dims"],
            "n_action_dims": cfg["n_action_dims"],
            "n_bits": n_bits,
            "bits_per_dim": bits_per_dim,
            "action_range": [cfg["action_low"], cfg["action_high"]],
        }

    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
