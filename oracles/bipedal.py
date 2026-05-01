#!/usr/bin/env python3
"""Gymnasium adapter for BipedalWalker-v3 with binary-weighted continuous actions.

State: 24D continuous.
Action: 4D continuous [-1, 1] (hip1, knee1, hip2, knee2 torques).

Binary decomposition: each action dimension uses k bits with weights {1, 2, 4}.
12 bit-predicates total, each a PredProg term.

Commands: collect_states, score_bit, score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import eval_pred, chain_action, run_oracle, score_batch_parallel

import numpy as np
import gymnasium as gym

NUM_DIMS = 24
BITS_PER_DIM = 3
N_ACTION_DIMS = 4
N_BITS = BITS_PER_DIM * N_ACTION_DIMS
MAX_STEPS = 1600
WEIGHTS = [2**i for i in range(BITS_PER_DIM)]
MAX_SUM = sum(WEIGHTS)


def bits_to_action(bit_values):
    actions = np.zeros(N_ACTION_DIMS)
    for d in range(N_ACTION_DIMS):
        s = 0
        for i in range(BITS_PER_DIM):
            s += WEIGHTS[i] * bit_values[d * BITS_PER_DIM + i]
        actions[d] = 2.0 * s / MAX_SUM - 1.0
    return actions


def bit_policy_action(bit_preds, obs):
    bits = [1 if eval_pred(p, obs) else 0 for p in bit_preds]
    return bits_to_action(bits)


def _run_episode_bits(args):
    bit_preds, seed, max_steps = args
    env = gym.make("BipedalWalker-v3")
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    for _ in range(max_steps):
        action = bit_policy_action(bit_preds, obs)
        obs, r, term, trunc, _ = env.step(action)
        total_r += r
        if term or trunc:
            break
    env.close()
    return total_r


def run_episodes_bits(bit_preds, seeds, max_steps=MAX_STEPS):
    total = 0.0
    survived = 0
    for s in seeds:
        r = _run_episode_bits((bit_preds, s, max_steps))
        total += r
        if r > 0:
            survived += 1
    return total, survived


def collect_states_bits(bit_preds, seeds, max_steps=MAX_STEPS):
    all_states = []
    n_survived = 0
    for seed in seeds:
        env = gym.make("BipedalWalker-v3")
        obs, _ = env.reset(seed=seed)
        ep_r = 0.0
        for _ in range(max_steps):
            all_states.append(obs.tolist())
            action = bit_policy_action(bit_preds, obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc:
                break
        env.close()
        if ep_r > 0:
            n_survived += 1
    return all_states, n_survived


def _score_bit_one(args):
    idx, candidate, bit_preds, target_bit, seeds, max_steps = args
    test_preds = list(bit_preds)
    test_preds[target_bit] = candidate
    reward, survived = run_episodes_bits(test_preds, seeds, max_steps)
    return {"idx": idx, "reward": reward, "landings": survived}


def score_bit_batch(candidates, bit_preds, target_bit, seeds,
                    max_steps=MAX_STEPS):
    args_list = [
        (i, cand, bit_preds, target_bit, seeds, max_steps)
        for i, cand in enumerate(candidates)
    ]
    return score_batch_parallel(_score_bit_one, args_list)


def _run_episode_chain(args):
    chain, default, seed, max_steps = args
    env = gym.make("BipedalWalker-v3")
    obs, _ = env.reset(seed=seed)
    ep_r = 0.0
    for _ in range(max_steps):
        a = chain_action(chain, default, obs)
        obs, r, term, trunc, _ = env.step(a)
        ep_r += r
        if term or trunc:
            break
    env.close()
    return ep_r


def dispatch(request):
    cmd = request["cmd"]

    if cmd == "collect_states":
        bit_preds = request.get("bit_predicates")
        if bit_preds is not None:
            seeds = request.get("seeds", list(range(40)))
            max_steps = request.get("max_steps", MAX_STEPS)
            states, n_survived = collect_states_bits(
                bit_preds, seeds, max_steps)
            return {"states": states, "n_landings": n_survived,
                    "n_episodes": len(seeds)}
        else:
            seeds = request.get("seeds", list(range(40)))
            max_steps = request.get("max_steps", MAX_STEPS)
            bp = ["falsep"] * N_BITS
            states, n_survived = collect_states_bits(bp, seeds, max_steps)
            return {"states": states, "n_landings": n_survived,
                    "n_episodes": len(seeds)}

    elif cmd == "score_bit":
        candidates = request["candidates"]
        bit_preds = request["bit_predicates"]
        target_bit = request["target_bit"]
        seeds = request.get("seeds", list(range(30)))
        max_steps = request.get("max_steps", MAX_STEPS)

        baseline_reward, baseline_survived = run_episodes_bits(
            bit_preds, seeds, max_steps)
        scores = score_bit_batch(
            candidates, bit_preds, target_bit, seeds, max_steps)
        return {"scores": scores, "baseline_reward": baseline_reward,
                "baseline_landings": baseline_survived}

    elif cmd == "score":
        candidates = request["candidates"]
        stage_action = request["stage_action"]
        default = request["default"]
        chain_so_far = [(p, a) for p, a in request.get("chain_so_far", [])]
        chain_after = [(p, a) for p, a in request.get("chain_after", [])]
        seeds = request.get("seeds", list(range(30)))
        max_steps = request.get("max_steps", MAX_STEPS)

        baseline_chain = chain_so_far + chain_after
        total = 0.0
        wins = 0
        for s in seeds:
            r = _run_episode_chain(
                (baseline_chain, default, s, max_steps))
            total += r
            if r > 0:
                wins += 1
        return {"scores": [], "baseline_reward": total,
                "baseline_landings": wins}
    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
