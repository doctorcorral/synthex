#!/usr/bin/env python3
"""Per-state episode-reward ranking oracle.

Given a deployed policy (full-ranking partition), this oracle:
1. Runs the policy on episodes, collecting (state, env_snapshot) pairs
2. At each sampled state, clones the env, tries each action, follows
   the deployed policy for the rest of the episode
3. Returns per-state rankings by episode reward

Supports both classic Gymnasium (replay-based) and ALE (state-cloning).

Commands:
    per_state_rank   — deploy policy, profile per-state rankings
    exhaustive_rank  — evaluate all possible full-ranking partitions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import eval_pred, run_oracle, score_batch_parallel

import numpy as np
import gymnasium as gym
from multiprocessing import Pool, cpu_count

# ── Environment registry ─────────────────────────────────────────

ENV_CONFIGS = {
    "lunarlander": {
        "gym_id": "LunarLander-v3",
        "num_dims": 6,
        "num_actions": 4,
        "is_ale": False,
        "max_steps": 1000,
        "action_map": None,
        "win_threshold": 100,
    },
    "cartpole": {
        "gym_id": "CartPole-v1",
        "num_dims": 4,
        "num_actions": 2,
        "is_ale": False,
        "max_steps": 500,
        "action_map": None,
        "win_threshold": 499,
    },
    "acrobot": {
        "gym_id": "Acrobot-v1",
        "num_dims": 6,
        "num_actions": 3,
        "is_ale": False,
        "max_steps": 500,
        "action_map": None,
        "win_threshold": -499,
    },
    "mountaincar": {
        "gym_id": "MountainCar-v0",
        "num_dims": 2,
        "num_actions": 3,
        "is_ale": False,
        "max_steps": 200,
        "action_map": None,
        "win_threshold": -199,
    },
    "pong": {
        "gym_id": "ALE/Pong-v5",
        "num_dims": 6,
        "num_actions": 3,
        "is_ale": True,
        "max_steps": 10000,
        "action_map": {0: 2, 1: 0, 2: 3},
        "win_threshold": 0,
        "obs_type": "ram",
    },
    "breakout": {
        "gym_id": "ALE/Breakout-v5",
        "num_dims": 5,
        "num_actions": 3,
        "is_ale": True,
        "max_steps": 10000,
        "action_map": {0: 3, 1: 0, 2: 2},
        "win_threshold": 0,
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
    return obs[:cfg["num_dims"]].tolist()


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


# ── Per-state profiling: Classic Gym (replay-based) ──────────────

def _rewards_at_step_replay(env_name, seed, actions_so_far, region_id,
                            preds_rankings, default_ranking, max_steps):
    cfg = ENV_CONFIGS[env_name]
    n_actions = cfg["num_actions"]
    rewards = [0.0] * n_actions

    for aid in range(n_actions):
        env = make_env(env_name)
        env.reset(seed=seed)
        for a in actions_so_far:
            env.step(a)
        obs, reward, term, trunc, _ = env.step(aid)
        cumulative = reward
        steps_left = max(0, max_steps - len(actions_so_far) - 1)
        for _ in range(steps_left):
            if term or trunc:
                break
            s = extract_state_classic(env, obs, cfg)
            a = partition_action_override(
                s, preds_rankings, default_ranking, region_id, aid)
            obs, reward, term, trunc, _ = env.step(a)
            cumulative += reward
        rewards[aid] = cumulative
        env.close()

    return rewards


def _profile_episode_classic(args):
    env_name, seed, preds_rankings, default_ranking, \
        max_steps, sample_interval = args
    cfg = ENV_CONFIGS[env_name]

    env = make_env(env_name)
    obs, _ = env.reset(seed=seed)
    actions_taken = []
    results = []

    for step in range(max_steps):
        state = extract_state_classic(env, obs, cfg)

        if step > 0 and step % sample_interval == 0:
            region = partition_region(state, preds_rankings)
            rewards = _rewards_at_step_replay(
                env_name, seed, actions_taken, region,
                preds_rankings, default_ranking, max_steps)
            results.append({
                "state": state,
                "rewards": rewards,
                "region": region
            })

        action = partition_action(state, preds_rankings, default_ranking)
        actions_taken.append(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    env.close()
    return results


# ── Per-state profiling: ALE (state-cloning) ─────────────────────

def _extract_state_ale(env, env_name, prev_bx, prev_by):
    if env_name == "pong":
        return extract_state_pong(env, prev_bx, prev_by)
    elif env_name == "breakout":
        return extract_state_breakout(env, prev_bx, prev_by)
    raise ValueError(f"Unknown ALE env: {env_name}")


def _rewards_at_state_ale(env, env_name, snapshot, region_id,
                          preds_rankings, default_ranking, max_steps,
                          prev_bx, prev_by):
    cfg = ENV_CONFIGS[env_name]
    n_actions = cfg["num_actions"]
    action_map = cfg["action_map"]
    rewards = [0.0] * n_actions

    for aid in range(n_actions):
        env.unwrapped.ale.restoreState(snapshot)
        ale_action = action_map[aid] if action_map else aid
        _, reward, term, trunc, _ = env.step(ale_action)
        cumulative = reward
        p_bx, p_by = prev_bx, prev_by

        for _ in range(max_steps):
            if term or trunc:
                break
            s, p_bx, p_by = _extract_state_ale(env, env_name, p_bx, p_by)
            a = partition_action_override(
                s, preds_rankings, default_ranking, region_id, aid)
            ale_a = action_map[a] if action_map else a
            _, reward, term, trunc, _ = env.step(ale_a)
            cumulative += reward

        rewards[aid] = cumulative

    return rewards


def _profile_episode_ale(args):
    env_name, seed, preds_rankings, default_ranking, \
        max_steps, sample_interval = args
    cfg = ENV_CONFIGS[env_name]
    action_map = cfg["action_map"]

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
            rewards = _rewards_at_state_ale(
                env, env_name, snapshot, region,
                preds_rankings, default_ranking,
                max_steps - step, prev_bx, prev_by)
            results.append({
                "state": state,
                "rewards": rewards,
                "region": region
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

def per_state_rank(env_name, preds_rankings, default_ranking,
                   seeds, max_steps=None, sample_interval=10):
    cfg = ENV_CONFIGS[env_name]
    if max_steps is None:
        max_steps = cfg["max_steps"]
    is_ale = cfg["is_ale"]

    if is_ale:
        profile_fn = _profile_episode_ale
    else:
        profile_fn = _profile_episode_classic

    args_list = [
        (env_name, seed, preds_rankings, default_ranking,
         max_steps, sample_interval)
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


def analyze_rankings(profile_data, n_actions, n_regions):
    from collections import Counter

    region_rewards = {}
    for i in range(n_regions):
        region_rewards[i] = {"rewards": [0.0] * n_actions, "count": 0,
                             "per_state_rankings": []}
    region_rewards[-1] = {"rewards": [0.0] * n_actions, "count": 0,
                          "per_state_rankings": []}

    for entry in profile_data:
        rid = entry["region"]
        rewards = entry["rewards"]
        region_rewards[rid]["count"] += 1
        for a in range(n_actions):
            region_rewards[rid]["rewards"][a] += rewards[a]
        per_state_ranking = sorted(range(n_actions),
                                   key=lambda a: rewards[a], reverse=True)
        region_rewards[rid]["per_state_rankings"].append(
            tuple(per_state_ranking))

    analysis = {}
    for region_id, data in region_rewards.items():
        n_states = data["count"]
        if n_states == 0:
            analysis[region_id] = {
                "n_states": 0,
                "ranking": None,
                "avg_rewards": [],
                "consistency": 0.0,
            }
            continue

        avg_rewards = [r / n_states for r in data["rewards"]]
        ranking = sorted(range(n_actions),
                         key=lambda a: avg_rewards[a], reverse=True)

        n_agree = sum(1 for psr in data["per_state_rankings"]
                      if list(psr) == ranking)
        consistency = round(n_agree / n_states * 100, 1)

        top_action_agree = sum(
            1 for psr in data["per_state_rankings"]
            if psr[0] == ranking[0])
        top_consistency = round(top_action_agree / n_states * 100, 1)

        counts = Counter(data["per_state_rankings"])
        top5 = {str(list(k)): v for k, v in counts.most_common(5)}

        analysis[region_id] = {
            "n_states": n_states,
            "ranking": ranking,
            "avg_rewards": [round(r, 2) for r in avg_rewards],
            "consistency": consistency,
            "top_action_consistency": top_consistency,
            "distribution": top5,
        }

    return analysis


# ── Exhaustive whole-policy evaluation ────────────────────────────

def _eval_policy_batch(args):
    env_name, preds_rankings, default_ranking, seeds, max_steps = args
    cfg = ENV_CONFIGS[env_name]
    total_reward = 0.0
    wins = 0

    for seed in seeds:
        env = make_env(env_name)
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0

        for _ in range(max_steps):
            state = extract_state_classic(env, obs, cfg)
            action = partition_action(state, preds_rankings, default_ranking)
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                break

        total_reward += ep_reward
        if ep_reward >= cfg.get("win_threshold", float('inf')):
            wins += 1
        env.close()

    return total_reward / len(seeds), wins


def exhaustive_rank(env_name, preds, top_actions, n_actions,
                    seeds, max_steps=None):
    from itertools import permutations, product

    cfg = ENV_CONFIGS[env_name]
    if max_steps is None:
        max_steps = cfg["max_steps"]

    n_regions = len(preds) + 1
    assert len(top_actions) == n_regions

    region_candidates = []
    for top in top_actions:
        rest = [a for a in range(n_actions) if a != top]
        candidates = tuple([top] + list(p) for p in permutations(rest))
        region_candidates.append(candidates)

    all_policies = list(product(*region_candidates))
    n_policies = len(all_policies)

    print(f"  Exhaustive search: {n_policies} candidate policies, "
          f"{len(seeds)} episodes each", flush=True)

    tasks = []
    for policy_combo in all_policies:
        pred_rankings = [(preds[i], list(policy_combo[i]))
                         for i in range(len(preds))]
        default_ranking = list(policy_combo[-1])
        tasks.append((env_name, pred_rankings, default_ranking,
                      seeds, max_steps))

    n_workers = min(cpu_count(), len(tasks))
    with Pool(processes=n_workers) as pool:
        results = pool.map(_eval_policy_batch, tasks)

    scored = []
    for i, (avg_reward, wins) in enumerate(results):
        scored.append({
            "policy": [list(r) for r in all_policies[i]],
            "avg_reward": round(avg_reward, 2),
            "wins": wins,
        })
    scored.sort(key=lambda x: x["avg_reward"], reverse=True)

    return scored


# ── Dispatch ─────────────────────────────────────────────────────

def dispatch(request):
    cmd = request["cmd"]

    if cmd == "per_state_rank":
        env_name = request["env"]
        preds_rankings = [
            (p["pred"], p["ranking"]) for p in request["preds"]]
        default_ranking = request["default_ranking"]
        seeds = request.get("seeds", list(range(20)))
        max_steps = request.get("max_steps", None)
        sample_interval = request.get("sample_interval", 10)

        profile_data = per_state_rank(
            env_name, preds_rankings, default_ranking,
            seeds, max_steps, sample_interval)

        cfg = ENV_CONFIGS[env_name]
        n_regions = len(preds_rankings)
        analysis = analyze_rankings(
            profile_data, cfg["num_actions"], n_regions)

        states = [e["state"] for e in profile_data]
        regions = [e["region"] for e in profile_data]

        return {
            "states": states,
            "regions": regions,
            "analysis": {str(k): v for k, v in analysis.items()},
            "n_profile_points": len(profile_data),
        }

    elif cmd == "exhaustive_rank":
        env_name = request["env"]
        preds = [p["pred"] for p in request["preds"]]
        top_actions = request["top_actions"]
        n_actions = request["n_actions"]
        seeds = request.get("seeds", list(range(50)))
        max_steps = request.get("max_steps", None)

        scored = exhaustive_rank(
            env_name, preds, top_actions, n_actions, seeds, max_steps)

        return {
            "top_policies": scored[:20],
            "n_evaluated": len(scored),
        }

    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
