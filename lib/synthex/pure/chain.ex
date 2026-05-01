defmodule Synthex.Pure.Chain do
  @moduledoc """
  Decision-chain synthesis for k-action environments.

  Instead of C(k,2) pairwise comparisons (prone to Condorcet cycles),
  builds a priority chain of k-1 binary predicates:

      if p1(s) then action1
      else if p2(s) then action2
      ...
      else default_action

  Each stage discovers one predicate by directly optimizing total
  trajectory reward.  No oracle consistency required -- just pick the
  predicate that makes the chain perform best.

  Cycles are impossible by construction.
  """

  alias Synthex.Core.{PredProg, ContinuousFeatures, CEGIS}

  # ── Chain policy evaluation ──────────────────────────────────────────

  def chain_action([], default, _state_list, _eval_fn), do: default
  def chain_action([{pred, action} | rest], default, state_list, eval_fn) do
    if PredProg.eval(pred, state_list, eval_fn) do
      action
    else
      chain_action(rest, default, state_list, eval_fn)
    end
  end

  # ── Reward-based rollout ─────────────────────────────────────────────

  def reward_rollout(_state, _policy_fn, _env_mod, 0), do: 0
  def reward_rollout(state, policy_fn, env_mod, remaining) do
    if env_mod.terminal?(state) do
      0
    else
      step_reward = max(0, env_mod.max_penalty() - env_mod.penalty(state))
      action = policy_fn.(state)
      s_prime = env_mod.step(state, action)
      step_reward + reward_rollout(s_prime, policy_fn, env_mod, remaining - 1)
    end
  end

  # ── Scoring ──────────────────────────────────────────────────────────

  def total_reward(chain, default, env_mod, eval_fn) do
    horizon = env_mod.score_horizon()
    policy_fn = fn state ->
      chain_action(chain, default, env_mod.state_to_list(state), eval_fn)
    end

    Enum.reduce(env_mod.starts(), 0, fn s0, acc ->
      acc + reward_rollout(s0, policy_fn, env_mod, horizon)
    end)
  end

  # ── Feature generation ──────────────────────────────────────────────

  defp generate_features(env_mod, eval_fn, chain_so_far, default, max_coeff) do
    actions = env_mod.actions()

    traj_states =
      Enum.flat_map(env_mod.starts(), fn s0 ->
        single_action_trajs =
          Enum.flat_map(actions, fn a ->
            collect_trajectory(s0, fn _s -> a end, env_mod, 40)
          end)

        chain_traj =
          if chain_so_far != [] do
            policy_fn = fn state ->
              chain_action(chain_so_far, default, env_mod.state_to_list(state), eval_fn)
            end
            collect_trajectory(s0, policy_fn, env_mod, 60)
          else
            []
          end

        single_action_trajs ++ chain_traj
      end)
      |> Enum.uniq()

    traj_lists = Enum.map(traj_states, &env_mod.state_to_list/1)
    state_size = length(env_mod.state_to_list(hd(env_mod.starts())))
    ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
  end

  # ── Per-stage synthesis ──────────────────────────────────────────────

  defp solve_stage(stage_action, default, chain_so_far, env_mod, eval_fn, opts) do
    depth = Keyword.get(opts, :depth, 0)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 3)

    IO.puts("\n  Stage: #{inspect(stage_action)} vs chain fallback")
    IO.puts("  Chain so far: #{length(chain_so_far)} predicates")
    IO.puts("  Depth: #{depth}, MaxCoeff: #{max_coeff}")

    baseline_reward = total_reward(chain_so_far, default, env_mod, eval_fn)
    IO.puts("  Baseline reward (no #{inspect(stage_action)}): #{baseline_reward}")

    cegar_feature_loop(stage_action, default, chain_so_far, env_mod, eval_fn, depth, max_coeff, cegar_rounds, nil)
  end

  defp cegar_feature_loop(_stage_action, _default, _chain_so_far, _env_mod, _eval_fn, _depth, _max_coeff, 0, best_so_far) do
    best_so_far
  end

  defp cegar_feature_loop(stage_action, default, chain_so_far, env_mod, eval_fn, depth, max_coeff, rounds_left, best_so_far) do
    IO.puts("\n  Feature expansion round #{4 - rounds_left}/3")

    effective_chain =
      case best_so_far do
        nil -> chain_so_far
        pred -> chain_so_far ++ [{pred, stage_action}]
      end

    features = generate_features(env_mod, eval_fn, effective_chain, default, max_coeff)
    candidates = CEGIS.enumerate(features, depth)
    IO.puts("  Features: #{length(features)}, Candidates: #{length(candidates)}")

    case score_candidates(candidates, stage_action, default, chain_so_far, env_mod, eval_fn) do
      nil ->
        IO.puts("  No improving candidate found")
        best_so_far

      {pred, reward} ->
        IO.puts("  Best: #{inspect(pred)}  reward=#{reward}")

        improved =
          case best_so_far do
            nil -> true
            _ ->
              old_chain = chain_so_far ++ [{best_so_far, stage_action}]
              old_reward = total_reward(old_chain, default, env_mod, eval_fn)
              reward > old_reward
          end

        if improved do
          cegar_feature_loop(stage_action, default, chain_so_far, env_mod, eval_fn, depth, max_coeff, rounds_left - 1, pred)
        else
          IO.puts("  No improvement over previous best")
          best_so_far
        end
    end
  end

  defp score_candidates(candidates, stage_action, default, chain_so_far, env_mod, eval_fn) do
    baseline_reward = total_reward(chain_so_far, default, env_mod, eval_fn)

    result =
      candidates
      |> Flow.from_enumerable(stages: System.schedulers_online())
      |> Flow.map(fn pred ->
        test_chain = chain_so_far ++ [{pred, stage_action}]
        reward = total_reward(test_chain, default, env_mod, eval_fn)
        {pred, reward}
      end)
      |> Enum.max_by(fn {_pred, reward} -> reward end, fn -> nil end)

    case result do
      nil -> nil
      {pred, reward} ->
        if reward > baseline_reward do
          {pred, reward}
        else
          nil
        end
    end
  end

  # ── Trajectory collection ───────────────────────────────────────────

  defp collect_trajectory(_state, _policy_fn, _env_mod, 0), do: []
  defp collect_trajectory(state, policy_fn, env_mod, n) do
    if env_mod.terminal?(state) do
      []
    else
      action = policy_fn.(state)
      s_prime = env_mod.step(state, action)
      [state | collect_trajectory(s_prime, policy_fn, env_mod, n - 1)]
    end
  end

  # ── Multi-stage orchestrator ─────────────────────────────────────────

  @doc """
  Synthesize a complete decision chain.

  ## Options
    - `:depth` -- boolean depth (default 0; try 1 if depth 0 fails)
    - `:max_coeff` -- max diagonal coefficient (default 5)
    - `:cegar_rounds` -- feature expansion rounds per stage (default 3)
  """
  def solve(env_mod, action_priority, default_action, opts \\ []) do
    eval_fn = &ContinuousFeatures.eval_feature/2

    IO.puts("  Chain Synthesis (reward-based)")
    IO.puts("  Environment: #{inspect(env_mod)}")
    IO.puts("  Priority: #{inspect(action_priority)} > #{inspect(default_action)}")
    IO.puts("  Options: #{inspect(opts)}\n")

    {final_chain, _} =
      Enum.reduce(action_priority, {[], 1}, fn action, {chain_so_far, stage_num} ->
        IO.puts("\n>> STAGE #{stage_num}/#{length(action_priority)}: #{inspect(action)}")

        case solve_stage(action, default_action, chain_so_far, env_mod, eval_fn, opts) do
          nil ->
            IO.puts("  No predicate found for #{inspect(action)}")
            {chain_so_far, stage_num + 1}

          pred ->
            new_chain = chain_so_far ++ [{pred, action}]
            reward = total_reward(new_chain, default_action, env_mod, eval_fn)
            IO.puts("  Chain reward after stage #{stage_num}: #{reward}")
            {new_chain, stage_num + 1}
        end
      end)

    IO.puts("\n  CHAIN SYNTHESIS COMPLETE")
    IO.puts("Final chain (#{length(final_chain)} predicates):")

    Enum.each(final_chain, fn {pred, action} ->
      IO.puts("  #{inspect(action)}  when  #{inspect(pred)}")
    end)

    IO.puts("  #{inspect(default_action)}  otherwise")

    final_reward = total_reward(final_chain, default_action, env_mod, eval_fn)
    baseline = total_reward([], default_action, env_mod, eval_fn)
    IO.puts("\nFinal reward: #{final_reward}  (baseline do-nothing: #{baseline})")

    {final_chain, default_action}
  end
end
