#!/usr/bin/env python3
"""Gymnasium adapter for CartPole-v1 with ranking synthesis support.

State: [x, x_dot, theta, theta_dot] (4D)
Actions: 0=Left, 1=Right
Reward: +1 per step, max 500 steps.

Commands: collect_states, score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import chain_action, run_oracle, score_batch_parallel

import gymnasium as gym

NUM_DIMS = 4
NUM_ACTIONS = 2
MAX_STEPS = 500


def _run_episode(args):
    chain, default, seed, max_steps = args
    env = gym.make("CartPole-v1")
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


def run_episodes(chain, default, seeds, max_steps=MAX_STEPS):
    total = 0.0
    wins = 0
    for s in seeds:
        r = _run_episode((chain, default, s, max_steps))
        total += r
        if r >= 475:
            wins += 1
    return total, wins


def collect_states(chain, default, seeds, max_steps=MAX_STEPS):
    all_states = []
    n_wins = 0
    for seed in seeds:
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=seed)
        ep_r = 0.0
        for _ in range(max_steps):
            all_states.append(obs.tolist())
            a = chain_action(chain, default, obs)
            obs, r, term, trunc, _ = env.step(a)
            ep_r += r
            if term or trunc:
                break
        env.close()
        if ep_r >= 475:
            n_wins += 1
    return all_states, n_wins


def _score_one(args):
    idx, candidate, chain_so_far, stage_action, default, seeds, \
        chain_after, max_steps = args
    test_chain = chain_so_far + [(candidate, stage_action)] + chain_after
    reward, wins = run_episodes(test_chain, default, seeds, max_steps)
    return {"idx": idx, "reward": reward, "landings": wins}


def score_batch(candidates, stage_action, default, chain_so_far, seeds,
                chain_after=None, max_steps=MAX_STEPS):
    if chain_after is None:
        chain_after = []
    args_list = [
        (i, cand, chain_so_far, stage_action, default, seeds,
         chain_after, max_steps)
        for i, cand in enumerate(candidates)
    ]
    return score_batch_parallel(_score_one, args_list)


def dispatch(request):
    cmd = request["cmd"]

    if cmd == "collect_states":
        chain = [(p, a) for p, a in request.get("chain", [])]
        default = request["default"]
        seeds = request.get("seeds", list(range(10)))
        max_steps = request.get("max_steps", MAX_STEPS)
        states, n_wins = collect_states(chain, default, seeds, max_steps)
        return {"states": states, "n_landings": n_wins, "n_episodes": len(seeds)}

    elif cmd == "score":
        candidates = request["candidates"]
        stage_action = request["stage_action"]
        default = request["default"]
        chain_so_far = [(p, a) for p, a in request.get("chain_so_far", [])]
        chain_after = [(p, a) for p, a in request.get("chain_after", [])]
        seeds = request.get("seeds", list(range(10)))
        max_steps = request.get("max_steps", MAX_STEPS)
        baseline_chain = chain_so_far + chain_after
        baseline_reward, baseline_wins = run_episodes(
            baseline_chain, default, seeds, max_steps)
        scores = score_batch(
            candidates, stage_action, default, chain_so_far, seeds,
            chain_after, max_steps)
        return {"scores": scores, "baseline_reward": baseline_reward,
                "baseline_landings": baseline_wins}
    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
