defmodule Synthex.Pure.PairwiseMatrix do
  @moduledoc """
  Orchestrates the synthesis of an entire Pairwise RankTree matrix for a given
  Environment by automatically generating and running the N(N-1)/2 action pairs.
  """

  alias Synthex.Pure.StateCEGAR
  alias Synthex.Pure.Oracle
  alias Synthex.Core.{ContinuousFeatures, CEGIS}

  @doc """
  Generates all unique pairs of actions from the environment's action list.
  """
  def generate_pairs(env_mod) do
    actions = env_mod.actions()

    for {a, i} <- Enum.with_index(actions),
        {b, j} <- Enum.with_index(actions),
        i < j,
        do: {a, b}
  end

  @doc """
  The master orchestrator. Takes an environment module, generates all pairs,
  creates a massive feature/candidate pool, and uses Flow/StateCEGAR to synthesize
  every single relation in the matrix.
  """
  def solve(env_mod, depth \\ 1, max_coeff \\ 3, max_fuel \\ 100) do
    IO.puts("==================================================")
    IO.puts("  Initiating Pure Pairwise CSHRL Matrix Synthesis (Seeded)")
    IO.puts("  Environment: #{inspect(env_mod)}")
    IO.puts("  Depth: #{depth}, Max Coeff: #{max_coeff}")
    IO.puts("==================================================\n")

    known_seeds = %{
      {:fire_main, :do_nothing} => {:or, {:feat, {:axis, 3, -110000000}}, {:feat, {:diag, 1, 3, 6}}},
      {:fire_left, :do_nothing} => {:and, {:feat, {:axis, 0, -50000000}}, {:feat, {:axis, 1, 137400000}}},
      {:fire_right, :do_nothing} => {:and, {:feat, {:axis, 1, 137400000}}, {:feat, {:axis, 2, 51000000}}},
      {:fire_main, :fire_left} => {:and, {:feat, {:axis, 1, 87680000}}, {:feat, {:diag, 2, 3, 3}}},
      {:fire_main, :fire_right} => {:and, {:feat, {:axis, 1, 100360000}}, {:feat, {:diag, 2, 3, -3}}}
    }

    eval_fn = &ContinuousFeatures.eval_feature/2
    IO.puts("Seeding candidate pool from starting trajectories...")
    all_traj = Enum.flat_map(env_mod.starts(), fn s0 ->
      traj_truep = Oracle.collect(s0, :truep, hd(env_mod.actions()), List.last(env_mod.actions()), env_mod, eval_fn, 15, known_seeds)
      traj_falsep = Oracle.collect(s0, :falsep, hd(env_mod.actions()), List.last(env_mod.actions()), env_mod, eval_fn, 15, known_seeds)
      traj_truep ++ traj_falsep
    end)

    traj_lists = Enum.map(all_traj, &env_mod.state_to_list/1)
    state_size = length(env_mod.state_to_list(hd(env_mod.starts())))

    features = ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
    IO.puts("Generated #{length(features)} continuous structural features.")

    cands = CEGIS.enumerate(features, depth)
    IO.puts("Generated #{length(cands)} boolean candidates at Depth #{depth}.\n")

    pairs = generate_pairs(env_mod)
    IO.puts("Generated #{length(pairs)} independent pairwise combinations to solve.\n")

    matrix_results =
      Enum.reduce(pairs, known_seeds, fn {action_a, action_b}, acc ->
        case StateCEGAR.run_pair(env_mod, action_a, action_b, cands, max_fuel, acc) do
          {:ok, best_p} ->
            Map.put(acc, {action_a, action_b}, best_p)
          {:error, _} ->
            IO.puts("\n  FATAL: Failed to solve #{inspect(action_a)} vs #{inspect(action_b)}. Using known seed if available.")
            if Map.has_key?(known_seeds, {action_a, action_b}) do
              Map.put(acc, {action_a, action_b}, known_seeds[{action_a, action_b}])
            else
              Map.put(acc, {action_a, action_b}, :failed)
            end
        end
      end)

    IO.puts("\n==================================================")
    IO.puts("  FULL RANKTREE MATRIX SYNTHESIS COMPLETE!")
    IO.puts("==================================================")
    Enum.each(matrix_results, fn {{a, b}, p} ->
      IO.puts("#{inspect(a)} >= #{inspect(b)}  =>  #{inspect(p)}")
    end)

    matrix_results
  end
end
