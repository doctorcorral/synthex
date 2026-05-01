#!/usr/bin/env python3
"""Gymnasium adapter for ALE/Tetris-v5 with ranking synthesis support (23D state).

Full board representation: 20 row bitmasks + piece info.

Dims 0-19:  board row bitmasks (RAM[0..19]), row 0 = top
Dim 20:     piece_x (horizontal position, RAM[106])
Dim 21:     piece_y (vertical position, RAM[105])
Dim 22:     rotation/type encoding (RAM[104])

Actions (synthesis → ALE):
  0=LEFT(3), 1=RIGHT(2), 2=ROTATE/FIRE(1), 3=DOWN(4), 4=NOOP(0)

Commands: collect_states, score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import chain_action, run_oracle, score_batch_parallel

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

NUM_DIMS = 23
NUM_ACTIONS = 5
ACTION_TO_ALE = {0: 3, 1: 2, 2: 1, 3: 4, 4: 0}
ACTION_NAMES = {0: "left", 1: "right", 2: "rotate", 3: "down", 4: "noop"}
MAX_STEPS = 5000


def extract_state(ram):
    board_rows = [float(ram[i]) for i in range(20)]
    piece_x = float(ram[106])
    piece_y = float(ram[105])
    rotation_raw = float(ram[104])
    return board_rows + [piece_x, piece_y, rotation_raw]


def _run_episode(args):
    chain, default, seed, max_steps = args
    env = gym.make("ALE/Tetris-v5", obs_type="ram")
    env.reset(seed=seed)

    prev_piece_count = 0
    prev_board_fill = 0
    lines_cleared = 0
    steps = 0

    for step in range(max_steps):
        ram = env.unwrapped.ale.getRAM()
        state = extract_state(ram)

        piece_count = int(ram[110])
        board_fill = int(sum(int(ram[i]) for i in range(20)))

        if piece_count > prev_piece_count and board_fill < prev_board_fill:
            lines_cleared += 1
        prev_piece_count = piece_count
        prev_board_fill = board_fill

        action_idx = chain_action(chain, default, state)
        ale_action = ACTION_TO_ALE[action_idx]
        _, _, term, trunc, _ = env.step(ale_action)
        steps += 1
        if term or trunc:
            break

    env.close()
    return float(steps + 100 * lines_cleared)


def run_episodes(chain, default, seeds, max_steps=MAX_STEPS):
    total = 0.0
    scoring = 0
    for s in seeds:
        r = _run_episode((chain, default, s, max_steps))
        total += r
        if r > 200:
            scoring += 1
    return total, scoring


def collect_states(chain, default, seeds, max_steps=MAX_STEPS):
    all_states = []
    n_scoring = 0
    for seed in seeds:
        env = gym.make("ALE/Tetris-v5", obs_type="ram")
        env.reset(seed=seed)
        steps = 0
        prev_piece_count = 0
        prev_board_fill = 0
        lines_cleared = 0
        for _ in range(max_steps):
            ram = env.unwrapped.ale.getRAM()
            state = extract_state(ram)
            all_states.append(state)

            piece_count = int(ram[110])
            board_fill = int(sum(int(ram[i]) for i in range(20)))
            if piece_count > prev_piece_count and board_fill < prev_board_fill:
                lines_cleared += 1
            prev_piece_count = piece_count
            prev_board_fill = board_fill

            action_idx = chain_action(chain, default, state)
            ale_action = ACTION_TO_ALE[action_idx]
            _, _, term, trunc, _ = env.step(ale_action)
            steps += 1
            if term or trunc:
                break
        env.close()
        ep_reward = steps + 100 * lines_cleared
        if ep_reward > 200:
            n_scoring += 1
    return all_states, n_scoring


def _score_one(args):
    idx, candidate, chain_so_far, stage_action, default, seeds, \
        chain_after, max_steps = args
    test_chain = chain_so_far + [(candidate, stage_action)] + chain_after
    reward, scoring = run_episodes(test_chain, default, seeds, max_steps)
    return {"idx": idx, "reward": reward, "landings": scoring}


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
        states, n_scoring = collect_states(chain, default, seeds, max_steps)
        return {
            "states": states,
            "n_landings": n_scoring,
            "n_episodes": len(seeds),
        }

    elif cmd == "score":
        candidates = request["candidates"]
        stage_action = request["stage_action"]
        default = request["default"]
        chain_so_far = [(p, a) for p, a in request.get("chain_so_far", [])]
        chain_after = [(p, a) for p, a in request.get("chain_after", [])]
        seeds = request.get("seeds", list(range(5)))
        max_steps = request.get("max_steps", MAX_STEPS)
        baseline_chain = chain_so_far + chain_after
        baseline_reward, baseline_scoring = run_episodes(
            baseline_chain, default, seeds, max_steps)
        scores = score_batch(
            candidates, stage_action, default, chain_so_far, seeds,
            chain_after, max_steps)
        return {
            "scores": scores,
            "baseline_reward": baseline_reward,
            "baseline_landings": baseline_scoring,
        }

    else:
        return {"error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    run_oracle(dispatch)
