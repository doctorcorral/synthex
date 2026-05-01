#!/usr/bin/env python3
"""Gymnasium adapter for LunarLander-v3 (chain-based scoring).

Simpler variant of the lunarlander oracle, focused on
collect_states and score with chain-action policies.

Commands: collect_states, score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import chain_action, run_oracle, score_batch_parallel

import gymnasium as gym

NUM_DIMS = 6


def collect_states(chain, default, seeds, max_steps=1000):
    env = gym.make("LunarLander-v3")
    all_states = []
    n_landings = 0

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        for _ in range(max_steps):
            state = obs[:NUM_DIMS].tolist()
            all_states.append(state)
            action = chain_action(chain, default, state)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        if ep_reward > 100:
            n_landings += 1

    env.close()
    return all_states, n_landings


def run_episodes(chain, default, seeds, max_steps=1000):
    env = gym.make("LunarLander-v3")
    total_reward = 0.0
    landings = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        for _ in range(max_steps):
            state = obs[:6].tolist()
            action = chain_action(chain, default, state)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_reward += ep_reward
        if ep_reward > 100:
            landings += 1
    env.close()
    return total_reward, landings


def _score_one(args):
    idx, candidate, chain_so_far, stage_action, default, seeds, chain_after, max_steps = args
    test_chain = chain_so_far + [(candidate, stage_action)] + chain_after
    reward, landings = run_episodes(test_chain, default, seeds, max_steps)
    return {"idx": idx, "reward": reward, "landings": landings}


def score_batch(candidates, stage_action, default, chain_so_far, seeds, chain_after=None, max_steps=1000):
    if chain_after is None:
        chain_after = []
    args_list = [
        (i, cand, chain_so_far, stage_action, default, seeds, chain_after, max_steps)
        for i, cand in enumerate(candidates)
    ]
    return score_batch_parallel(_score_one, args_list)


def dispatch(request):
    cmd = request["cmd"]

    if cmd == "collect_states":
        chain = [(p, a) for p, a in request.get("chain", [])]
        default = request["default"]
        seeds = request.get("seeds", list(range(40)))
        max_steps = request.get("max_steps", 300)
        states, n_land = collect_states(chain, default, seeds, max_steps)
        return {"states": states, "n_landings": n_land, "n_episodes": len(seeds)}

    elif cmd == "score":
        candidates = request["candidates"]
        stage_action = request["stage_action"]
        default = request["default"]
        chain_so_far = [(p, a) for p, a in request.get("chain_so_far", [])]
        chain_after = [(p, a) for p, a in request.get("chain_after", [])]
        seeds = request.get("seeds", list(range(30)))
        max_steps = request.get("max_steps", 1000)
        baseline_chain = chain_so_far + chain_after
        baseline_reward, baseline_landings = run_episodes(baseline_chain, default, seeds, max_steps)
        scores = score_batch(candidates, stage_action, default, chain_so_far, seeds, chain_after, max_steps)
        return {
            "scores": scores,
            "baseline_reward": baseline_reward,
            "baseline_landings": baseline_landings,
        }

    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
