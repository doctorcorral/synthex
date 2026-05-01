defmodule Synthex.Pure.Permutation.StateCEGAR do
  @moduledoc """
  A formal, rigorous synthesis engine for Permutation Policies.
  Aligns with the pure CSHRL methodology: it uses structural self-play rollouts 
  to evaluate candidates, ensuring that the synthesized policy is coinductively 
  self-consistent.

  To avoid combinatorial explosion, it builds the sequence of conditional swaps
  greedily. At each step, it finds the single CondSwap that minimizes the true 
  rollout penalty across the counter-example anchors.
  """

  alias Synthex.Pure.Permutation.Policy
  alias Synthex.Pure.Permutation.Oracle

  @doc """
  Computes the sum of rollout penalties for a given policy across all anchor states.
  """
  def evaluate_policy(pol, anchors, env_mod, eval_fn) do
    Enum.reduce(anchors, 0, fn state, acc ->
      acc + penalty_rollout_substeps(state, pol, env_mod, eval_fn, env_mod.oracle_horizon())
    end)
  end

  defp penalty_rollout_substeps(_state, _pol, _env_mod, _eval_fn, 0), do: 0
  defp penalty_rollout_substeps(state, pol, env_mod, eval_fn, k) do
    if env_mod.terminal?(state) do
      env_mod.penalty(state)
    else
      action = Policy.best_action(pol, env_mod.state_to_list(state), eval_fn)
      {s_prime, accumulated_penalty} = apply_substeps(state, action, env_mod, 5, 0)
      accumulated_penalty + penalty_rollout_substeps(s_prime, pol, env_mod, eval_fn, k - 1)
    end
  end

  defp apply_substeps(state, _action, _env_mod, 0, acc_penalty), do: {state, acc_penalty}
  defp apply_substeps(state, action, env_mod, steps_left, acc_penalty) do
    if env_mod.terminal?(state) do
      {state, acc_penalty}
    else
      s_prime = env_mod.step(state, action)
      apply_substeps(s_prime, action, env_mod, steps_left - 1, acc_penalty + env_mod.penalty(state))
    end
  end

  @doc """
  Greedily finds the single best Base Ranking for the current anchors.
  """
  def find_best_base(anchors, actions, env_mod, eval_fn) do
    base_rankings = Synthex.Pure.Permutation.Engine.permutations(actions)
    
    base_rankings
    |> Task.async_stream(fn base ->
      pol = %Policy{base_ranking: base, swaps: []}
      score = evaluate_policy(pol, anchors, env_mod, eval_fn)
      {base, score}
    end, ordered: false)
    |> Enum.map(fn {:ok, res} -> res end)
    |> Enum.min_by(fn {_base, score} -> score end)
  end

  @doc """
  Greedily finds the best CondSwap to append to the current policy.
  """
  def find_best_swap(current_pol, anchors, cands, swaps, env_mod, eval_fn) do
    current_score = evaluate_policy(current_pol, anchors, env_mod, eval_fn)
    
    best_cand =
      cands
      |> Flow.from_enumerable(stages: System.schedulers_online())
      |> Flow.map(fn pred ->
        # For a given predicate, test all possible swap operations
        swaps
        |> Enum.map(fn op ->
          cond_swap = {pred, op}
          new_pol = %Policy{base_ranking: current_pol.base_ranking, swaps: current_pol.swaps ++ [cond_swap]}
          score = evaluate_policy(new_pol, anchors, env_mod, eval_fn)
          {cond_swap, score}
        end)
        |> Enum.min_by(fn {_cs, score} -> score end)
      end)
      |> Enum.min_by(fn {_cs, score} -> score end, fn -> nil end)

    case best_cand do
      nil -> nil
      {best_cs, new_score} ->
        if new_score < current_score do
          {best_cs, new_score}
        else
          # No swap improves the policy
          nil
        end
    end
  end

  @doc """
  Iteratively builds a Permutation Policy up to `max_swaps`.
  """
  def build_policy(anchors, actions, cands, swaps, env_mod, eval_fn, max_swaps) do
    {best_base, base_score} = find_best_base(anchors, actions, env_mod, eval_fn)
    initial_pol = %Policy{base_ranking: best_base, swaps: []}
    
    IO.puts("   Best Base Ranking Score: #{base_score}")
    
    expand_policy(initial_pol, anchors, cands, swaps, env_mod, eval_fn, max_swaps, 0)
  end

  defp expand_policy(pol, _anchors, _cands, _swaps, _env_mod, _eval_fn, max_swaps, step) when step >= max_swaps, do: pol
  defp expand_policy(pol, anchors, cands, swaps, env_mod, eval_fn, max_swaps, step) do
    case find_best_swap(pol, anchors, cands, swaps, env_mod, eval_fn) do
      nil ->
        IO.puts("   No further swap improves the policy. Stopping sequence early.")
        pol
      {best_cs, new_score} ->
        IO.puts("   Added Swap #{step + 1}: Score improved to #{new_score}")
        new_pol = %Policy{base_ranking: pol.base_ranking, swaps: pol.swaps ++ [best_cs]}
        expand_policy(new_pol, anchors, cands, swaps, env_mod, eval_fn, max_swaps, step + 1)
    end
  end

  @doc """
  Collects a trajectory using the policy and returns states that have high penalty (crashes).
  """
  def collect_cegar_states(_state, _pol, _env_mod, _eval_fn, 0), do: []
  def collect_cegar_states(state, pol, env_mod, eval_fn, n) do
    current_penalty = env_mod.penalty(state)
    # Threshold for a "hard" state to be added as a counter-example
    is_hard = current_penalty > 1000

    if env_mod.terminal?(state) do
      if is_hard, do: [state], else: []
    else
      action = Policy.best_action(pol, env_mod.state_to_list(state), eval_fn)
      s_prime = env_mod.step(state, action)
      
      rest = collect_cegar_states(s_prime, pol, env_mod, eval_fn, n - 1)
      
      if is_hard do
        [state | rest]
      else
        rest
      end
    end
  end

  @doc """
  The Outer Loop: Synthesizes a policy, tests it for counter-examples, and repeats.
  """
  def solve(env_mod, depth \\ 1, max_coeff \\ 1, max_swaps \\ 4, cegar_iters \\ 5) do
    IO.puts("==================================================")
    IO.puts("🛠️  Initiating Pure Structural StateCEGAR (Permutations)")
    IO.puts("Environment: #{inspect(env_mod)}")
    IO.puts("Max Swaps: #{max_swaps}, Depth: #{depth}, Iterations: #{cegar_iters}")
    IO.puts("==================================================\n")

    eval_fn = &Synthex.Core.ContinuousFeatures.eval_feature/2
    starts = env_mod.starts()
    actions = env_mod.actions()
    swaps = Synthex.Pure.Permutation.Engine.generate_swaps(length(actions))

    cegar_loop(env_mod, starts, actions, swaps, eval_fn, depth, max_coeff, max_swaps, cegar_iters, 1)
  end

  defp cegar_loop(env_mod, anchors, actions, swaps, eval_fn, depth, max_coeff, max_swaps, max_iters, iter) do
    IO.puts("\n--- CEGAR Iteration #{iter}/#{max_iters} ---")
    IO.puts("1. Anchors: #{length(anchors)} states")

    # Generate trajectories from anchors to build features
    traj_lists = Enum.map(anchors, &env_mod.state_to_list/1)
    state_size = length(hd(traj_lists))

    features = Synthex.Core.ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
    cands = Synthex.Core.CEGIS.enumerate(features, depth)
    
    # Prune predicates that don't differentiate any anchor states
    pruned_cands = Enum.filter(cands, fn pred ->
      evals = Enum.map(traj_lists, fn sl -> Synthex.Core.PredProg.eval(pred, sl, eval_fn) end)
      Enum.any?(evals, &(&1 == true)) and Enum.any?(evals, &(&1 == false))
    end)

    IO.puts("2. Predicates: #{length(pruned_cands)} active candidates (Depth #{depth})")
    IO.puts("3. Synthesizing Greedy Swap Sequence...")

    pol = build_policy(anchors, actions, pruned_cands, swaps, env_mod, eval_fn, max_swaps)

    if iter >= max_iters do
      IO.puts("\n==================================================")
      IO.puts("🏆 FINAL STRUCTURAL POLICY")
      IO.puts("==================================================")
      IO.inspect(pol)
      pol
    else
      IO.puts("4. Structural Rollout Validation (Finding CEXs)...")
      new_states = Enum.flat_map(env_mod.starts(), fn start ->
        collect_cegar_states(start, pol, env_mod, eval_fn, 150)
      end)
      
      all_new_states = new_states |> Enum.uniq()
      
      sampled_new = if length(all_new_states) > 50 do
        Enum.take_random(all_new_states, 50)
      else
        all_new_states
      end
      
      all_anchors = (anchors ++ sampled_new) |> Enum.uniq()

      if length(all_anchors) == length(anchors) do
        IO.puts("   No new counter-examples found. Pure CEGAR Converged!")
        IO.puts("\n==================================================")
        IO.puts("🏆 FINAL STRUCTURAL POLICY")
        IO.puts("==================================================")
        IO.inspect(pol)
        pol
      else
        IO.puts("   Discovered #{length(all_anchors) - length(anchors)} new counter-examples.")
        cegar_loop(env_mod, all_anchors, actions, swaps, eval_fn, depth, max_coeff, max_swaps, max_iters, iter + 1)
      end
    end
  end
end
