#!/usr/bin/env python3
"""Pairwise action comparison oracle for propagation-grounded Phase 2.

For each pair of actions (a,b), at sampled states along the deployed
policy trajectory, commits to each action persistently for the region
and compares cumulative episode rewards.

Returns PredObs-compatible observations: (state, a_wins: bool) for
each pair, suitable for CEGIS predicate synthesis.

Commands:
    pairwise_compare — deploy policy, collect pairwise observations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import eval_pred, run_oracle

import numpy as np
import gymnasium as gym
from multiprocessing import Pool, cpu_count
from itertools import combinations

# ── Environment registry ─────────────────────────────────────────

ENV_CONFIGS = {
    "lunarlander": {
        "gym_id": "LunarLander-v3",
        "num_dims": 6,
        "num_actions": 4,
        "is_ale": False,
        "max_steps": 1000,
        "action_map": None,
    },
    "cartpole": {
        "gym_id": "CartPole-v1",
        "num_dims": 4,
        "num_actions": 2,
        "is_ale": False,
        "max_steps": 500,
        "action_map": None,
    },
    "acrobot": {
        "gym_id": "Acrobot-v1",
        "num_dims": 6,
        "num_actions": 3,
        "is_ale": False,
        "max_steps": 500,
        "action_map": None,
    },
    "mountaincar": {
        "gym_id": "MountainCar-v0",
        "num_dims": 2,
        "num_actions": 3,
        "is_ale": False,
        "max_steps": 200,
        "action_map": None,
    },
    "pong": {
        "gym_id": "ALE/Pong-v5",
        "num_dims": 6,
        "num_actions": 3,
        "is_ale": True,
        "max_steps": 10000,
        "action_map": {0: 2, 1: 0, 2: 3},
        "obs_type": "ram",
    },
    "breakout": {
        "gym_id": "ALE/Breakout-v5",
        "num_dims": 5,
        "num_actions": 3,
        "is_ale": True,
        "max_steps": 10000,
        "action_map": {0: 3, 1: 0, 2: 2},
        "obs_type": "ram",
    },
}


def make_env(env_name):
    cfg = ENV_CONFIGS[env_name]
    kwargs = {}
    if cfg.get("obs_type"):
        kwargs["obs_type"] = cfg["obs_type"]
    return gym.make(cfg["gym_id"], **kwargs)


# ── State extraction ─────────────────────────────────────────────

def extract_state_classic(env, obs, cfg):
    return [float(x) for x in obs[:cfg["num_dims"]]]


def extract_state_pong(env, prev_bx, prev_by):
    ram = env.unwrapped.ale.getRAM()
    bx, by = float(ram[49]), float(ram[54])
    py, ey = float(ram[51]), float(ram[50])
    if prev_by is not None and by > 0 and by < 200 and prev_by > 0:
        vx, vy = bx - prev_bx, by - prev_by
    else:
        vx, vy = 0.0, 0.0
    return [bx, by, py, vx, vy, ey], bx, by


def extract_state_breakout(env, prev_bx, prev_by):
    ram = env.unwrapped.ale.getRAM()
    bx, by, px = float(ram[99]), float(ram[101]), float(ram[72])
    if prev_by is not None:
        dx, dy = bx - prev_bx, by - prev_by
    else:
        dx, dy = 0.0, 0.0
    return [bx, by, px, dx, dy], bx, by


def _extract_state_ale(env, env_name, prev_bx, prev_by):
    if env_name == "pong":
        return extract_state_pong(env, prev_bx, prev_by)
    elif env_name == "breakout":
        return extract_state_breakout(env, prev_bx, prev_by)
    raise ValueError(f"Unknown ALE env: {env_name}")


# ── Policy evaluation ────────────────────────────────────────────

def partition_action(state, preds_rankings, default_ranking):
    for pred, ranking in preds_rankings:
        if eval_pred(pred, state):
            return ranking[0]
    return default_ranking[0]


def partition_action_override(state, preds_rankings, default_ranking,
                              override_region, override_action):
    for i, (pred, ranking) in enumerate(preds_rankings):
        if eval_pred(pred, state):
            if i == override_region:
                return override_action
            return ranking[0]
    if override_region == -1:
        return override_action
    return default_ranking[0]


def partition_region(state, preds_rankings):
    for i, (pred, _ranking) in enumerate(preds_rankings):
        if eval_pred(pred, state):
            return i
    return -1


# ── Pairwise comparison: Classic Gym (replay-based) ──────────────

def _compare_pair_replay(env_name, seed, actions_so_far, region_id,
                         action_a, action_b,
                         preds_rankings, default_ranking, max_steps):
    cfg = ENV_CONFIGS[env_name]
    rewards = {}

    for candidate in [action_a, action_b]:
        env = make_env(env_name)
        env.reset(seed=seed)
        for a in actions_so_far:
            env.step(a)
        obs, reward, term, trunc, _ = env.step(candidate)
        cumulative = reward
        steps_left = max(0, max_steps - len(actions_so_far) - 1)
        for _ in range(steps_left):
            if term or trunc:
                break
            s = extract_state_classic(env, obs, cfg)
            a = partition_action_override(
                s, preds_rankings, default_ranking, region_id, candidate)
            obs, reward, term, trunc, _ = env.step(a)
            cumulative += reward
        rewards[candidate] = cumulative
        env.close()

    return rewards[action_a], rewards[action_b]


def _pairwise_episode_classic(args):
    env_name, seed, preds_rankings, default_ranking, \
        max_steps, sample_interval, n_actions = args
    cfg = ENV_CONFIGS[env_name]
    all_pairs = list(combinations(range(n_actions), 2))

    env = make_env(env_name)
    obs, _ = env.reset(seed=seed)
    actions_taken = []
    results = []

    for step in range(max_steps):
        state = extract_state_classic(env, obs, cfg)

        if step > 0 and step % sample_interval == 0:
            region = partition_region(state, preds_rankings)
            for (a, b) in all_pairs:
                r_a, r_b = _compare_pair_replay(
                    env_name, seed, actions_taken, region,
                    a, b, preds_rankings, default_ranking, max_steps)
                results.append({
                    "state": state,
                    "region": int(region),
                    "pair": [int(a), int(b)],
                    "a_wins": bool(r_a > r_b),
                    "reward_a": float(round(r_a, 2)),
                    "reward_b": float(round(r_b, 2)),
                })

        action = partition_action(state, preds_rankings, default_ranking)
        actions_taken.append(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    env.close()
    return results


# ── Pairwise comparison: ALE (state-cloning) ─────────────────────

def _compare_pair_ale(env, env_name, snapshot, region_id,
                      action_a, action_b,
                      preds_rankings, default_ranking, max_steps,
                      prev_bx, prev_by):
    cfg = ENV_CONFIGS[env_name]
    action_map = cfg["action_map"]
    rewards = {}

    for candidate in [action_a, action_b]:
        env.unwrapped.ale.restoreState(snapshot)
        ale_action = action_map[candidate] if action_map else candidate
        _, reward, term, trunc, _ = env.step(ale_action)
        cumulative = reward
        p_bx, p_by = prev_bx, prev_by

        for _ in range(max_steps):
            if term or trunc:
                break
            s, p_bx, p_by = _extract_state_ale(env, env_name, p_bx, p_by)
            a = partition_action_override(
                s, preds_rankings, default_ranking, region_id, candidate)
            ale_a = action_map[a] if action_map else a
            _, reward, term, trunc, _ = env.step(ale_a)
            cumulative += reward

        rewards[candidate] = cumulative

    return rewards[action_a], rewards[action_b]


def _pairwise_episode_ale(args):
    env_name, seed, preds_rankings, default_ranking, \
        max_steps, sample_interval, n_actions = args
    cfg = ENV_CONFIGS[env_name]
    action_map = cfg["action_map"]
    all_pairs = list(combinations(range(n_actions), 2))

    env = make_env(env_name)
    env.reset(seed=seed)
    prev_bx, prev_by = None, None
    results = []

    for step in range(max_steps):
        state, prev_bx, prev_by = _extract_state_ale(
            env, env_name, prev_bx, prev_by)

        if step > 0 and step % sample_interval == 0:
            region = partition_region(state, preds_rankings)
            snapshot = env.unwrapped.ale.cloneState()
            for (a, b) in all_pairs:
                r_a, r_b = _compare_pair_ale(
                    env, env_name, snapshot, region,
                    a, b, preds_rankings, default_ranking,
                    max_steps - step, prev_bx, prev_by)
                results.append({
                    "state": state,
                    "region": int(region),
                    "pair": [int(a), int(b)],
                    "a_wins": bool(r_a > r_b),
                    "reward_a": float(round(r_a, 2)),
                    "reward_b": float(round(r_b, 2)),
                })
            env.unwrapped.ale.restoreState(snapshot)

        action = partition_action(state, preds_rankings, default_ranking)
        ale_action = action_map[action] if action_map else action
        _, reward, term, trunc, _ = env.step(ale_action)
        if term or trunc:
            break

    env.close()
    return results


# ── Main entry point ─────────────────────────────────────────────

def pairwise_compare(env_name, preds_rankings, default_ranking,
                     seeds, max_steps=None, sample_interval=10):
    cfg = ENV_CONFIGS[env_name]
    if max_steps is None:
        max_steps = cfg["max_steps"]
    is_ale = cfg["is_ale"]
    n_actions = cfg["num_actions"]

    if is_ale:
        profile_fn = _pairwise_episode_ale
    else:
        profile_fn = _pairwise_episode_classic

    args_list = [
        (env_name, seed, preds_rankings, default_ranking,
         max_steps, sample_interval, n_actions)
        for seed in seeds
    ]

    n = len(args_list)
    n_workers = min(cpu_count(), n)

    if is_ale or n <= 2:
        all_results = [profile_fn(a) for a in args_list]
    else:
        with Pool(processes=n_workers) as pool:
            all_results = pool.map(profile_fn, args_list)

    flat = []
    for episode_results in all_results:
        flat.extend(episode_results)
    return flat


# ── Dispatch ─────────────────────────────────────────────────────

def dispatch(request):
    cmd = request["cmd"]

    if cmd == "pairwise_compare":
        env_name = request["env"]
        preds_rankings = [
            (p["pred"], p["ranking"]) for p in request["preds"]]
        default_ranking = request["default_ranking"]
        seeds = request.get("seeds", list(range(50)))
        max_steps = request.get("max_steps", None)
        sample_interval = request.get("sample_interval", 10)

        observations = pairwise_compare(
            env_name, preds_rankings, default_ranking,
            seeds, max_steps, sample_interval)

        return {
            "observations": observations,
            "n_observations": len(observations),
        }

    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
