defmodule Synthex.Pure.StateCEGAR do
  @moduledoc """
  The scalable Inner (CEGIS) and Outer (CEGAR) loop engine, backed by Flow for
  safe parallel evaluation of millions of candidates.
  """

  alias Synthex.Pure.Oracle
  alias Synthex.Core.PredProg

  @doc "Checks if a predicate perfectly matches the oracle on all anchor states, ignoring ties."
  def is_consistent_on?(p, anchors, action_a, action_b, env_mod, eval_fn, known_pairs) do
    Enum.all?(anchors, fn s ->
      ans = Oracle.oracle_predict(p, s, action_a, action_b, env_mod, eval_fn, known_pairs)
      if ans == :tie do
        true
      else
        PredProg.eval(p, env_mod.state_to_list(s), eval_fn) == ans
      end
    end)
  end

  @doc "A predicate is viable if it is consistent on anchors AND has a non-zero sum score."
  def viable_on?(p, anchors, action_a, action_b, env_mod, eval_fn, known_pairs) do
    if Oracle.sum_score(p, action_a, action_b, env_mod, eval_fn, known_pairs) == 0 do
      false
    else
      is_consistent_on?(p, anchors, action_a, action_b, env_mod, eval_fn, known_pairs)
    end
  end

  @doc """
  The Inner Loop: Filters the massive candidate pool down to the single best
  predicate that satisfies the current anchors, using Flow for parallel map-reduce.
  """
  def solve_inner(cands, anchors, action_a, action_b, env_mod, eval_fn, known_pairs) do
    cands
    |> Flow.from_enumerable(stages: System.schedulers_online())
    |> Flow.filter(fn p -> viable_on?(p, anchors, action_a, action_b, env_mod, eval_fn, known_pairs) end)
    |> Enum.max_by(fn p -> Oracle.sum_score(p, action_a, action_b, env_mod, eval_fn, known_pairs) end, fn -> nil end)
  end

  @doc """
  Rolls out the best candidate across all starting states. If it finds a state where
  the candidate contradicts the Oracle, it returns that state as a Counterexample.
  """
  def find_cex(p, action_a, action_b, env_mod, eval_fn, known_pairs) do
    Enum.find_value(env_mod.starts(), nil, fn s0 ->
      traj = Oracle.collect(s0, p, action_a, action_b, env_mod, eval_fn, 50, known_pairs)
      Enum.find(traj, fn s ->
        ans = Oracle.oracle_predict(p, s, action_a, action_b, env_mod, eval_fn, known_pairs)
        ans != :tie and PredProg.eval(p, env_mod.state_to_list(s), eval_fn) != ans
      end)
    end)
  end

  @doc """
  The Outer Loop: Recursively asks the Inner Loop for a candidate. If the candidate
  has a counterexample, it is added to the anchors and the loop repeats.
  """
  def outer_loop(0, _anchors, _cands, _action_a, _action_b, _env_mod, _eval_fn, _known_pairs) do
    IO.puts("  Out of fuel in outer loop (StateCEGAR failed to converge)")
    {:error, :cegar_needed}
  end

  def outer_loop(fuel, anchors, cands, action_a, action_b, env_mod, eval_fn, known_pairs) do
    IO.puts("  Outer loop (fuel: #{fuel}), anchor count: #{length(anchors)}")

    case solve_inner(cands, anchors, action_a, action_b, env_mod, eval_fn, known_pairs) do
      nil ->
        IO.puts("  Inner loop failed (No candidate works for current anchors). Increase Depth/MaxCoeff.")
        {:error, :cegar_needed}

      best_p ->
        case find_cex(best_p, action_a, action_b, env_mod, eval_fn, known_pairs) do
          nil ->
            IO.puts("  Converged StateCEGAR for pair #{inspect(action_a)} vs #{inspect(action_b)}!")
            {:ok, best_p}

          cex ->
            IO.puts("  Found CEX, adding to anchors and repeating...")
            outer_loop(fuel - 1, [cex | anchors], cands, action_a, action_b, env_mod, eval_fn, known_pairs)
        end
    end
  end

  @doc "Entry point to solve a specific independent Pairwise relation."
  def run_pair(env_mod, action_a, action_b, cands, max_fuel \\ 100, known_pairs \\ %{}) do
    eval_fn = &Synthex.Core.ContinuousFeatures.eval_feature/2
    IO.puts("\n  Starting Pairwise Synthesis: #{inspect(action_a)} vs #{inspect(action_b)}")
    outer_loop(max_fuel, env_mod.starts(), cands, action_a, action_b, env_mod, eval_fn, known_pairs)
  end
end
