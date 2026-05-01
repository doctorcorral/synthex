#!/usr/bin/env python3
"""Gymnasium adapter for ALE/Tetris-v5 with ranking synthesis support (8D state).

State vector (8D): piece_x, piece_y, rotation_raw, piece_vis,
                   board_fill, bottom_fill, occupied_rows, pieces_placed

Actions (synthesis → ALE):
  0=LEFT(3), 1=RIGHT(2), 2=ROTATE/FIRE(1), 3=DOWN(4), 4=NOOP(0)

Commands: collect_states, successor_explore, score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import chain_action, run_oracle, score_batch_parallel

import gymnasium as gym
import ale_py
from multiprocessing import Pool, cpu_count

gym.register_envs(ale_py)

NUM_DIMS = 8
NUM_ACTIONS = 5
ACTION_TO_ALE = {0: 3, 1: 2, 2: 1, 3: 4, 4: 0}
ACTION_NAMES = {0: "left", 1: "right", 2: "rotate", 3: "down", 4: "noop"}
MAX_STEPS = 5000


def extract_state(ram):
    piece_x = float(ram[106])
    piece_y = float(ram[105])
    rotation_raw = float(ram[104])
    piece_vis = float(ram[111])

    board_bytes = [float(ram[i]) for i in range(20)]
    board_fill = sum(board_bytes)
    bottom_fill = sum(board_bytes[10:])
    occupied_rows = float(sum(1 for b in board_bytes if b > 0))
    pieces_placed = float(ram[110])

    return [
        piece_x, piece_y, rotation_raw, piece_vis,
        board_fill, bottom_fill, occupied_rows, pieces_placed,
    ]


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


def _successor_explore_one(args):
    chain, default, seed, max_steps, lookahead, sample_every = args
    env = gym.make("ALE/Tetris-v5", obs_type="ram")
    env.reset(seed=seed)
    ale = env.unwrapped.ale

    steps_data = []
    prev_pc, prev_bf = 0, 0
    total_lines = 0
    steps = 0

    for step_i in range(max_steps):
        ram = ale.getRAM()
        state = extract_state(ram)
        chosen = chain_action(chain, default, state)

        if step_i % sample_every == 0:
            saved = ale.cloneState()
            alt_rewards = {}

            for aidx in range(NUM_ACTIONS):
                ale.restoreState(saved)
                ale.act(ACTION_TO_ALE[aidx])
                rollout_steps = 1
                rollout_lines = 0

                if not ale.game_over():
                    r_pc = int(ale.getRAM()[110])
                    r_bf = sum(int(ale.getRAM()[i]) for i in range(20))

                    for _ in range(lookahead - 1):
                        r_ram = ale.getRAM()
                        r_state = extract_state(r_ram)
                        new_pc = int(r_ram[110])
                        new_bf = sum(int(r_ram[i]) for i in range(20))
                        if new_pc > r_pc and new_bf < r_bf:
                            rollout_lines += 1
                        r_pc, r_bf = new_pc, new_bf

                        r_act = chain_action(chain, default, r_state)
                        ale.act(ACTION_TO_ALE[r_act])
                        rollout_steps += 1
                        if ale.game_over():
                            break

                alt_rewards[str(aidx)] = float(rollout_steps + 100 * rollout_lines)

            steps_data.append([state, chosen, alt_rewards])
            ale.restoreState(saved)

        ale_action = ACTION_TO_ALE[chosen]
        _, _, term, trunc, _ = env.step(ale_action)

        ram2 = ale.getRAM()
        pc = int(ram2[110])
        bf = sum(int(ram2[i]) for i in range(20))
        if pc > prev_pc and bf < prev_bf:
            total_lines += 1
        prev_pc, prev_bf = pc, bf
        steps += 1

        if term or trunc:
            break

    env.close()
    ep_reward = float(steps + 100 * total_lines)
    return steps_data, ep_reward


def successor_explore(chain, default, seeds, max_steps=MAX_STEPS,
                      lookahead=100, sample_every=10):
    args_list = [(chain, default, s, max_steps, lookahead, sample_every)
                 for s in seeds]
    n_workers = min(cpu_count(), len(args_list), 8)
    if len(args_list) <= 2:
        results = [_successor_explore_one(a) for a in args_list]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_successor_explore_one, args_list)

    all_steps = []
    n_scoring = 0
    for steps_data, ep_reward in results:
        all_steps.extend(steps_data)
        if ep_reward > 200:
            n_scoring += 1
    return all_steps, n_scoring


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

    elif cmd == "successor_explore":
        chain = [(p, a) for p, a in request.get("chain", [])]
        default = request["default"]
        seeds = request.get("seeds", list(range(10)))
        max_steps = request.get("max_steps", MAX_STEPS)
        lookahead = request.get("lookahead", 100)
        sample_every = request.get("sample_every", 10)
        steps_data, n_scoring = successor_explore(
            chain, default, seeds, max_steps, lookahead, sample_every)
        return {
            "steps": steps_data,
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
