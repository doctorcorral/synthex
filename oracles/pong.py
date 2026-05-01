#!/usr/bin/env python3
"""Gymnasium adapter for ALE/Pong-v5 with ranking synthesis support.

Uses RAM observations to extract a compact state vector:
  [ball_x, ball_y, player_y, ball_vx, ball_vy, enemy_y]

Actions: UP(2), NOOP(0), DOWN(3)
  Mapped to synthesis indices: 0=UP, 1=NOOP, 2=DOWN

Commands: collect_states, score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import chain_action, run_oracle, score_batch_parallel

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

NUM_DIMS = 6
NUM_ACTIONS = 3
ACTION_TO_ALE = {0: 2, 1: 0, 2: 3}
ACTION_NAMES = {0: "up", 1: "noop", 2: "down"}
MAX_STEPS_PER_EPISODE = 10000


def extract_state(env, prev_ball_x, prev_ball_y):
    ram = env.unwrapped.ale.getRAM()
    bx = float(ram[49])
    by = float(ram[54])
    py = float(ram[51])
    ey = float(ram[50])

    if prev_ball_y is not None and by > 0 and by < 200 and prev_ball_y > 0:
        vx = bx - prev_ball_x
        vy = by - prev_ball_y
    else:
        vx, vy = 0.0, 0.0

    return [bx, by, py, vx, vy, ey], bx, by


def collect_states(chain, default, seeds, max_steps=MAX_STEPS_PER_EPISODE):
    all_states = []
    n_wins = 0
    for seed in seeds:
        env = gym.make("ALE/Pong-v5", obs_type="ram")
        env.reset(seed=seed)
        prev_bx, prev_by = None, None
        ep_reward = 0.0
        for _ in range(max_steps):
            state, prev_bx, prev_by = extract_state(env, prev_bx, prev_by)
            all_states.append(state)
            action_idx = chain_action(chain, default, state)
            ale_action = ACTION_TO_ALE[action_idx]
            _, reward, term, trunc, _ = env.step(ale_action)
            ep_reward += reward
            if term or trunc:
                break
        env.close()
        if ep_reward > 0:
            n_wins += 1
    return all_states, n_wins


def _run_episode(args):
    chain, default, seed, max_steps = args
    env = gym.make("ALE/Pong-v5", obs_type="ram")
    env.reset(seed=seed)
    prev_bx, prev_by = None, None
    ep_reward = 0.0
    for _ in range(max_steps):
        state, prev_bx, prev_by = extract_state(env, prev_bx, prev_by)
        action_idx = chain_action(chain, default, state)
        ale_action = ACTION_TO_ALE[action_idx]
        _, reward, term, trunc, _ = env.step(ale_action)
        ep_reward += reward
        if term or trunc:
            break
    env.close()
    return ep_reward


def run_episodes(chain, default, seeds, max_steps=MAX_STEPS_PER_EPISODE):
    total = 0.0
    wins = 0
    for s in seeds:
        r = _run_episode((chain, default, s, max_steps))
        total += r
        if r > 0:
            wins += 1
    return total, wins


def _score_one(args):
    idx, candidate, chain_so_far, stage_action, default, seeds, \
        chain_after, max_steps = args
    test_chain = chain_so_far + [(candidate, stage_action)] + chain_after
    reward, wins = run_episodes(test_chain, default, seeds, max_steps)
    return {"idx": idx, "reward": reward, "landings": wins}


def score_batch(candidates, stage_action, default, chain_so_far, seeds,
                chain_after=None, max_steps=MAX_STEPS_PER_EPISODE):
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
        max_steps = request.get("max_steps", MAX_STEPS_PER_EPISODE)
        states, n_wins = collect_states(chain, default, seeds, max_steps)
        return {
            "states": states,
            "n_landings": n_wins,
            "n_episodes": len(seeds),
        }

    elif cmd == "score":
        candidates = request["candidates"]
        stage_action = request["stage_action"]
        default = request["default"]
        chain_so_far = [(p, a) for p, a in request.get("chain_so_far", [])]
        chain_after = [(p, a) for p, a in request.get("chain_after", [])]
        seeds = request.get("seeds", list(range(10)))
        max_steps = request.get("max_steps", MAX_STEPS_PER_EPISODE)
        baseline_chain = chain_so_far + chain_after
        baseline_reward, baseline_wins = run_episodes(
            baseline_chain, default, seeds, max_steps)
        scores = score_batch(
            candidates, stage_action, default, chain_so_far, seeds,
            chain_after, max_steps)
        return {
            "scores": scores,
            "baseline_reward": baseline_reward,
            "baseline_landings": baseline_wins,
        }

    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
