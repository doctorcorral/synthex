defmodule Synthex.Pure.Permutation.Engine do
  @moduledoc """
  The core synthesis engine for Permutation Policies.
  Instead of CEGAR on N(N-1)/2 pairs, this explores the joint space of
  Base Rankings and Conditional Swaps, using Flow to massively parallelize
  the rollout evaluations.
  """

  alias Synthex.Pure.Permutation.Policy
  alias Synthex.Pure.Permutation.Oracle

  @doc """
  Generates all permutations of a given list.
  Used to generate the initial Base Rankings.
  """
  def permutations([]), do: [[]]
  def permutations(list) do
    for elem <- list,
        rest <- permutations(list -- [elem]),
        do: [elem | rest]
  end

  @doc """
  Generates all valid swap operations for a given list size.
  """
  def generate_swaps(num_actions) do
    for i <- 0..(num_actions - 1),
        j <- 0..(num_actions - 1),
        i < j,
        do: {i, j}
  end

  @doc """
  Generates the initial candidate pool of Permutation Policies.
  Right now, we generate single-swap policies to keep the space manageable.
  """
  def generate_policies(env_mod, cands) do
    actions = env_mod.actions()
    num_actions = length(actions)
    
    base_rankings = permutations(actions)
    swaps = generate_swaps(num_actions)
    
    # 1. Policies with no swaps (just base rankings)
    no_swap_pols = Enum.map(base_rankings, fn base -> 
      %Policy{base_ranking: base, swaps: []} 
    end)
    
    # 2. Policies with exactly 1 conditional swap
    single_swap_pols = for base <- base_rankings,
                           swap <- swaps,
                           cond_prog <- cands do
      %Policy{base_ranking: base, swaps: [{cond_prog, swap}]}
    end

    no_swap_pols ++ single_swap_pols
  end

  @doc """
  The main entry point. Evaluates all generated policies and returns the one
  with the absolute lowest penalty score.
  """
  def solve(env_mod, depth \\ 1, max_coeff \\ 1) do
    IO.puts("==================================================")
    IO.puts("🚀 Initiating Permutation Policy Synthesis")
    IO.puts("Environment: #{inspect(env_mod)}")
    IO.puts("==================================================\n")

    eval_fn = &Synthex.Core.ContinuousFeatures.eval_feature/2
    
    # 1. Generate trajectory points (random sampling of the state space for features)
    IO.puts("Seeding candidate pool from starting trajectories...")
    all_traj = Enum.flat_map(env_mod.starts(), fn s0 ->
      traj_truep = Synthex.Pure.Oracle.collect(s0, :truep, hd(env_mod.actions()), List.last(env_mod.actions()), env_mod, eval_fn, 30, %{})
      traj_falsep = Synthex.Pure.Oracle.collect(s0, :falsep, hd(env_mod.actions()), List.last(env_mod.actions()), env_mod, eval_fn, 30, %{})
      traj_truep ++ traj_falsep
    end)

    traj_lists = Enum.map(all_traj, &env_mod.state_to_list/1)
    state_size = length(env_mod.state_to_list(hd(env_mod.starts())))

    features = Synthex.Core.ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
    IO.puts("Generated #{length(features)} continuous structural features.")

    cands = Synthex.Core.CEGIS.enumerate(features, depth)
    IO.puts("Generated #{length(cands)} boolean predicates at Depth #{depth}.\n")

    # 2. Construct the massive policy space
    policies = generate_policies(env_mod, cands)
    IO.puts("Generated #{length(policies)} Permutation Policies (Base Rankings + 1 Swap).")
    IO.puts("Evaluating rollouts concurrently...\n")

    # 3. Parallel Evaluation
    # We use Flow to evaluate the sum_penalty of every policy.
    # The policy with the lowest penalty wins.
    best_policy_with_score =
      policies
      |> Flow.from_enumerable(stages: System.schedulers_online())
      |> Flow.map(fn pol -> 
           score = Oracle.sum_penalty(pol, env_mod, eval_fn)
           {pol, score}
         end)
      |> Enum.min_by(fn {_pol, score} -> score end, fn -> nil end)

    case best_policy_with_score do
      nil -> 
        IO.puts("🚨 FATAL: Synthesis failed (empty candidate pool).")
        :error
      {best_pol, best_score} ->
        IO.puts("\n==================================================")
        IO.puts("🏆 PERMUTATION SYNTHESIS COMPLETE!")
        IO.puts("==================================================")
        IO.puts("Best Score (Penalty): #{best_score}")
        IO.inspect(best_pol)
        best_pol
    end
  end
end
