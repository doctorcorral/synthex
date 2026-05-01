defmodule Synthex.Pure.Permutation.OracleEngine do
  @moduledoc """
  The new Oracle-Guided Synthesis Engine for Permutation Policies.
  Instead of evaluating full rollouts for every permutation candidate (which
  causes combinatorial explosion), this engine treats synthesis as a supervised
  classification problem.

  1. Uses the Pairwise Oracle to generate a dataset of (State -> True Optimal Ranking).
  2. Generates structural continuous features.
  3. Synthesizes a `PermPolicy` (Base + N Swaps) that minimizes ranking errors
     against the ground truth dataset using Beam Search.
  """

  alias Synthex.Pure.Permutation.Policy

  @doc """
  Finds the true optimal ranking for a given state by rolling out 
  each action and scoring its penalty.
  """
  def get_true_ranking(state, env_mod, current_policy \\ nil) do
    actions = env_mod.actions()
    
    ranked_actions =
      Enum.map(actions, fn a ->
        penalty = simulate_action_penalty(state, a, env_mod, env_mod.oracle_horizon(), current_policy)
        {a, penalty}
      end)
      |> Enum.sort_by(fn {_a, penalty} -> penalty end)
      |> Enum.map(fn {a, _penalty} -> a end)

    ranked_actions
  end

  defp simulate_action_penalty(state, action, env_mod, _horizon, nil) do
    s_prime = env_mod.step(state, action)
    short_horizon = 10
    num_rollouts = 5
    
    total_random_penalty = Enum.reduce(1..num_rollouts, 0, fn _, acc ->
      acc + random_rollout(s_prime, env_mod, short_horizon)
    end)
    
    avg_random_penalty = div(total_random_penalty, num_rollouts)
    env_mod.penalty(state) + avg_random_penalty
  end

  defp random_rollout(_state, _env_mod, 0), do: 0
  defp random_rollout(state, env_mod, k) do
    if env_mod.terminal?(state) do
      env_mod.penalty(state)
    else
      random_action = Enum.random(env_mod.actions())
      s_prime = env_mod.step(state, random_action)
      env_mod.penalty(state) + random_rollout(s_prime, env_mod, k - 1)
    end
  end

  defp simulate_action_penalty(state, action, env_mod, _horizon, policy) do
    s_prime = env_mod.step(state, action)
    short_horizon = 50
    env_mod.penalty(state) + Synthex.Pure.Permutation.Oracle.penalty_rollout(s_prime, policy, env_mod, &Synthex.Core.ContinuousFeatures.eval_feature/2, short_horizon)
  end

  @doc """
  Computes distance between two rankings (e.g. Kendall tau distance proxy)
  """
  def ranking_error(predicted, target) do
    Enum.reduce(target, 0, fn act, acc ->
      pred_idx = Enum.find_index(predicted, &(&1 == act))
      targ_idx = Enum.find_index(target, &(&1 == act))
      acc + abs(pred_idx - targ_idx)
    end)
  end

  def evaluate_policy_error(pol, dataset, eval_fn) do
    Enum.reduce(dataset, 0, fn {_s, sl, true_rank}, acc ->
      pred_rank = Policy.evaluate(pol, sl, eval_fn)
      acc + ranking_error(pred_rank, true_rank)
    end)
  end

  @doc """
  Beam search to efficiently find the best sequence of conditional swaps.
  """
  def beam_search(dataset, base_rankings, pruned_cands, swaps, eval_fn, max_k, beam_width) do
    initial_beam = 
      Enum.map(base_rankings, fn base ->
        pol = %Policy{base_ranking: base, swaps: []}
        error = evaluate_policy_error(pol, dataset, eval_fn)
        {pol, error}
      end) 
      |> Enum.sort_by(fn {_, err} -> err end)
      |> Enum.take(beam_width)

    expand_beam(initial_beam, dataset, pruned_cands, swaps, eval_fn, 0, max_k, beam_width)
  end

  defp expand_beam(beam, _dataset, _cands, _swaps, _eval_fn, k, max_k, _beam_width) when k >= max_k, do: beam
  defp expand_beam(beam, dataset, cands, swaps, eval_fn, k, max_k, beam_width) do
    best_err = beam |> hd() |> elem(1)
    if best_err == 0 do
      beam
    else
      IO.puts("    Beam step #{k + 1}/#{max_k}... Best error so far: #{best_err}")
      
      expanded =
        beam
        |> Task.async_stream(fn {%Policy{base_ranking: base, swaps: current_swaps}, _err} ->
          for pred <- cands, op <- swaps do
            new_swaps = current_swaps ++ [{pred, op}]
            pol = %Policy{base_ranking: base, swaps: new_swaps}
            error = evaluate_policy_error(pol, dataset, eval_fn)
            {pol, error}
          end
        end, timeout: :infinity, ordered: false)
        |> Enum.flat_map(fn {:ok, results} -> results end)

      all_candidates = beam ++ expanded
      
      new_beam =
        all_candidates
        |> Enum.uniq_by(fn {%Policy{base_ranking: b, swaps: s}, _} -> {b, s} end)
        |> Enum.sort_by(fn {_, err} -> err end)
        |> Enum.take(beam_width)

      expand_beam(new_beam, dataset, cands, swaps, eval_fn, k + 1, max_k, beam_width)
    end
  end

  @doc """
  Main synthesis loop with CEGAR dataset expansion.
  """
  def solve(env_mod, depth \\ 1, max_coeff \\ 1, k_swaps \\ 4, cegar_iters \\ 4) do
    IO.puts("==================================================")
    IO.puts("🔮 Initiating StateCEGAR Oracle-Guided Permutation")
    IO.puts("Environment: #{inspect(env_mod)}")
    IO.puts("Max Swaps: #{k_swaps}, Depth: #{depth}, Iterations: #{cegar_iters}")
    IO.puts("==================================================\n")

    eval_fn = &Synthex.Core.ContinuousFeatures.eval_feature/2
    starts = env_mod.starts()
    
    # Initial states
    initial_states = Enum.flat_map(starts, fn s -> 
      [s, env_mod.step(s, hd(env_mod.actions()))] 
    end) |> Enum.uniq()

    cegar_loop(env_mod, initial_states, eval_fn, depth, max_coeff, k_swaps, cegar_iters, 1)
  end

  defp cegar_loop(env_mod, states, eval_fn, depth, max_coeff, k_swaps, max_iters, iter, best_pol \\ nil) do
    IO.puts("\n--- CEGAR Iteration #{iter}/#{max_iters} ---")
    IO.puts("1. Oracle Labeling #{length(states)} states...")
    
    dataset = Enum.map(states, fn s ->
      true_rank = get_true_ranking(s, env_mod, best_pol)
      {s, env_mod.state_to_list(s), true_rank}
    end)

    IO.puts("2. Generating and Pruning Predicates...")
    traj_lists = Enum.map(dataset, fn {_, sl, _} -> sl end)
    state_size = length(hd(traj_lists))

    features = Synthex.Core.ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
    cands = Synthex.Core.CEGIS.enumerate(features, depth)
    
    pruned_cands = Enum.filter(cands, fn pred ->
      evals = Enum.map(dataset, fn {_, sl, _} -> Synthex.Core.PredProg.eval(pred, sl, eval_fn) end)
      Enum.any?(evals, &(&1 == true)) and Enum.any?(evals, &(&1 == false))
    end)
    
    IO.puts("   Total predicates at Depth #{depth}: #{length(cands)}")
    IO.puts("   Pruned active predicates: #{length(pruned_cands)}")

    beam_width = 100
    IO.puts("3. Beam Search Synthesis (Beam Width: #{beam_width})...")
    actions = env_mod.actions()
    base_rankings = Synthex.Pure.Permutation.Engine.permutations(actions)
    swaps = Synthex.Pure.Permutation.Engine.generate_swaps(length(actions))

    beam = beam_search(dataset, base_rankings, pruned_cands, swaps, eval_fn, k_swaps, beam_width)
    {best_pol, best_err} = hd(beam)
    
    IO.puts("   Best Classification Error on Dataset: #{best_err}")

    if iter >= max_iters do
      IO.puts("\n==================================================")
      IO.puts("🏆 FINAL ORACLE-GUIDED POLICY")
      IO.puts("==================================================")
      IO.inspect(best_pol)
      best_pol
    else
      IO.puts("4. CEGAR Rollout Validation...")
      # Collect new states using the synthesized policy
      new_states = Enum.flat_map(env_mod.starts(), fn start ->
        collect_cegar_states(start, best_pol, env_mod, eval_fn, 50)
      end)
      
      # Keep states that seem risky or have high penalties, or just append uniquely
      all_new_states = new_states |> Enum.uniq()
      
      # Add max 100 new states per iteration to prevent predicate explosion
      sampled_new = if length(all_new_states) > 100 do
        Enum.take_random(all_new_states, 100)
      else
        all_new_states
      end
      
      all_states = (states ++ sampled_new) |> Enum.uniq()
      
      if length(all_states) == length(states) do
        IO.puts("   No new states discovered. CEGAR Converged!")
        IO.puts("\n==================================================")
        IO.puts("🏆 FINAL ORACLE-GUIDED POLICY")
        IO.puts("==================================================")
        IO.inspect(best_pol)
        best_pol
      else
        IO.puts("   Discovered #{length(all_states) - length(states)} new counter-example states.")
        cegar_loop(env_mod, all_states, eval_fn, depth, max_coeff, k_swaps, max_iters, iter + 1, best_pol)
      end
    end
  end

  defp collect_cegar_states(_state, _pol, _env_mod, _eval_fn, 0), do: []
  defp collect_cegar_states(state, pol, env_mod, eval_fn, n) do
    current_penalty = env_mod.penalty(state)
    is_hard = current_penalty > 1000

    if env_mod.terminal?(state) do
      if is_hard, do: [state], else: []
    else
      ranked_list = Policy.evaluate(pol, env_mod.state_to_list(state), eval_fn)
      action = hd(ranked_list)
      s_prime = env_mod.step(state, action)
      
      rest = collect_cegar_states(s_prime, pol, env_mod, eval_fn, n - 1)
      
      if is_hard do
        [state | rest]
      else
        rest
      end
    end
  end
end
