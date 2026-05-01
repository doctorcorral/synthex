#!/usr/bin/env python3
"""Gymnasium adapter for ALE/Breakout-v5 with ranking synthesis support.

Uses RAM observations to extract a compact state vector:
  [ball_x, ball_y, paddle_x, ball_dx, ball_dy]

Actions (synthesis): 0=RIGHT, 1=NOOP, 2=LEFT
ALE mapping: RIGHT=2, NOOP=0, LEFT=3, FIRE=1

Commands: collect_states, score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import chain_action, run_oracle, score_batch_parallel

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

NUM_DIMS = 5
NUM_ACTIONS = 3
ACTION_TO_ALE = {0: 2, 1: 0, 2: 3}
MAX_STEPS = 5000


def extract_state(env, prev_bx=None, prev_by=None):
    ram = env.unwrapped.ale.getRAM()
    bx = float(ram[99])
    by = float(ram[101])
    px = float(ram[72])
    if prev_bx is not None and by > 0:
        dx = bx - prev_bx
        if dx > 128: dx -= 256
        elif dx < -128: dx += 256
        dy = by - prev_by
        if dy > 128: dy -= 256
        elif dy < -128: dy += 256
    else:
        dx, dy = 0.0, 0.0
    return [bx, by, px, float(dx), float(dy)]


def _run_episode(args):
    chain, default, seed, max_steps = args
    env = gym.make("ALE/Breakout-v5", obs_type="ram")
    env.reset(seed=seed)
    ep_r = 0.0
    prev_bx, prev_by = None, None
    for _ in range(max_steps):
        state = extract_state(env, prev_bx, prev_by)
        prev_bx, prev_by = state[0], state[1]
        if state[1] == 0:
            ale_action = 1
        else:
            a = chain_action(chain, default, state)
            ale_action = ACTION_TO_ALE[a]
        _, r, term, trunc, _ = env.step(ale_action)
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
        if r > 0:
            wins += 1
    return total, wins


def collect_states(chain, default, seeds, max_steps=MAX_STEPS):
    all_states = []
    n_wins = 0
    for seed in seeds:
        env = gym.make("ALE/Breakout-v5", obs_type="ram")
        env.reset(seed=seed)
        ep_r = 0.0
        prev_bx, prev_by = None, None
        for _ in range(max_steps):
            state = extract_state(env, prev_bx, prev_by)
            prev_bx, prev_by = state[0], state[1]
            if state[1] > 0:
                all_states.append(state)
            if state[1] == 0:
                ale_action = 1
            else:
                a = chain_action(chain, default, state)
                ale_action = ACTION_TO_ALE[a]
            _, r, term, trunc, _ = env.step(ale_action)
            ep_r += r
            if term or trunc:
                break
        env.close()
        if ep_r > 0:
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
        seeds = request.get("seeds", list(range(5)))
        max_steps = request.get("max_steps", MAX_STEPS)
        states, n_wins = collect_states(chain, default, seeds, max_steps)
        return {"states": states, "n_landings": n_wins, "n_episodes": len(seeds)}

    elif cmd == "score":
        candidates = request["candidates"]
        stage_action = request["stage_action"]
        default = request["default"]
        chain_so_far = [(p, a) for p, a in request.get("chain_so_far", [])]
        chain_after = [(p, a) for p, a in request.get("chain_after", [])]
        seeds = request.get("seeds", list(range(5)))
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
