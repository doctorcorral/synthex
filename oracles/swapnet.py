#!/usr/bin/env python3
"""Gymnasium adapter for swap-network action ranking synthesis.

Ranking is produced by a sorting network: a fixed sequence of swap positions,
each conditionally applied based on a binary predicate over the state.

Commands:
    collect_states  — run episodes with current swap predicates, return states
    score_swap      — score candidate predicates for one swap position
    validate        — run episodes and return total reward
    search_base     — exhaustive search over base ranking permutations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import eval_pred, run_oracle, score_batch_parallel

import gymnasium as gym
from multiprocessing import Pool, cpu_count


def bubble_sort_network(n, passes=1):
    single = []
    for pass_idx in range(n - 1):
        for i in range(n - 1 - pass_idx):
            single.append((i, i + 1))
    return single * passes


def apply_swap_network(base_ranking, network, swap_preds, state):
    ranking = list(base_ranking)
    for (i, j), pred in zip(network, swap_preds):
        if eval_pred(pred, state):
            ranking[i], ranking[j] = ranking[j], ranking[i]
    return ranking


def swapnet_action(base_ranking, network, swap_preds, state):
    ranking = apply_swap_network(base_ranking, network, swap_preds, state)
    return ranking[0]


def _run_episode(args):
    env_name, base_ranking, network, swap_preds, seed, max_steps, n_dims = args
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    for _ in range(max_steps):
        state = obs[:n_dims].tolist()
        action = swapnet_action(base_ranking, network, swap_preds, state)
        obs, r, term, trunc, _ = env.step(action)
        total_r += r
        if term or trunc:
            break
    env.close()
    return total_r


def run_episodes(env_name, base_ranking, network, swap_preds, seeds,
                 max_steps, n_dims, parallel=False):
    args_list = [(env_name, base_ranking, network, swap_preds,
                  s, max_steps, n_dims) for s in seeds]
    if parallel and len(args_list) > 8:
        n_workers = min(cpu_count(), len(args_list), 8)
        with Pool(processes=n_workers) as pool:
            results = pool.map(_run_episode, args_list,
                               chunksize=max(1, len(args_list) // (n_workers * 2)))
    else:
        results = [_run_episode(a) for a in args_list]
    total = sum(results)
    successes = sum(1 for r in results if r > 100)
    return total, successes


def collect_states(env_name, base_ranking, network, swap_preds, seeds,
                   max_steps, n_dims):
    all_states = []
    successes = 0
    for seed in seeds:
        env = gym.make(env_name)
        obs, _ = env.reset(seed=seed)
        ep_r = 0.0
        for _ in range(max_steps):
            state = obs[:n_dims].tolist()
            all_states.append(state)
            action = swapnet_action(base_ranking, network, swap_preds, state)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc:
                break
        env.close()
        if ep_r > 100:
            successes += 1
    return all_states, successes


def _score_swap_one(args):
    (idx, candidate, env_name, base_ranking, network, swap_preds,
     target_idx, seeds, max_steps, n_dims) = args
    test_preds = list(swap_preds)
    test_preds[target_idx] = candidate
    reward, successes = run_episodes(
        env_name, base_ranking, network, test_preds, seeds, max_steps, n_dims)
    return {"idx": idx, "reward": reward, "landings": successes}


def score_swap_batch(candidates, env_name, base_ranking, network, swap_preds,
                     target_idx, seeds, max_steps, n_dims):
    args_list = [
        (i, cand, env_name, base_ranking, network, swap_preds,
         target_idx, seeds, max_steps, n_dims)
        for i, cand in enumerate(candidates)
    ]
    return score_batch_parallel(_score_swap_one, args_list)


def dispatch(request):
    cmd = request["cmd"]

    env_name = request.get("env_name", "LunarLander-v3")
    n_actions = request.get("n_actions", 4)
    n_dims = request.get("n_dims", 6)
    base_ranking = request.get("base_ranking", list(range(n_actions)))
    network = [tuple(p) for p in request.get("network",
                                              bubble_sort_network(n_actions))]
    swap_preds = request.get("swap_predicates",
                             ["falsep"] * len(network))
    seeds = request.get("seeds", list(range(30)))
    max_steps = request.get("max_steps", 300)

    if cmd == "collect_states":
        states, n_succ = collect_states(
            env_name, base_ranking, network, swap_preds, seeds,
            max_steps, n_dims)
        return {"states": states, "n_landings": n_succ,
                "n_episodes": len(seeds)}

    elif cmd == "score_swap":
        candidates = request["candidates"]
        target_idx = request["target_idx"]
        baseline_reward, baseline_succ = run_episodes(
            env_name, base_ranking, network, swap_preds, seeds,
            max_steps, n_dims, parallel=True)
        scores = score_swap_batch(
            candidates, env_name, base_ranking, network, swap_preds,
            target_idx, seeds, max_steps, n_dims)
        return {"scores": scores, "baseline_reward": baseline_reward,
                "baseline_landings": baseline_succ}

    elif cmd == "validate":
        reward, succ = run_episodes(
            env_name, base_ranking, network, swap_preds, seeds,
            max_steps, n_dims, parallel=True)
        return {"reward": reward, "landings": succ}

    elif cmd == "search_base":
        from itertools import permutations
        best_reward = -1e9
        best_ranking = base_ranking
        best_succ = 0
        for perm in permutations(range(n_actions)):
            r, s = run_episodes(
                env_name, list(perm), network,
                ["falsep"] * len(network), seeds, max_steps, n_dims,
                parallel=True)
            if r > best_reward:
                best_reward = r
                best_ranking = list(perm)
                best_succ = s
        return {"best_ranking": best_ranking, "reward": best_reward,
                "landings": best_succ}

    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
