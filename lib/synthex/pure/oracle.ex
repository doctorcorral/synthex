defmodule Synthex.Pure.Oracle do
  @moduledoc """
  A generic pairwise oracle that evaluates rollouts for any environment.
  It answers the question: "Is action_a better than action_b?"
  """

  alias Synthex.Core.PredProg

  @doc "Evaluates the tournament action using all known pairs."
  def tournament_action(state, p, action_a, action_b, known_pairs, env_mod, eval_fn) do
    actions = env_mod.actions()
    default_act = if :do_nothing in actions, do: :do_nothing, else: hd(actions)

    Enum.reduce(actions, default_act, fn a, best_so_far ->
      if a == best_so_far do
        best_so_far
      else
        beats? = check_beats(state, a, best_so_far, p, action_a, action_b, known_pairs, env_mod, eval_fn)
        if beats?, do: a, else: best_so_far
      end
    end)
  end

  defp check_beats(state, a, b, p, action_a, action_b, known_pairs, env_mod, eval_fn) do
    cond do
      {a, b} == {action_a, action_b} -> PredProg.eval(p, env_mod.state_to_list(state), eval_fn)
      {b, a} == {action_a, action_b} -> not PredProg.eval(p, env_mod.state_to_list(state), eval_fn)
      Map.has_key?(known_pairs, {a, b}) and known_pairs[{a, b}] != :failed ->
        PredProg.eval(known_pairs[{a, b}], env_mod.state_to_list(state), eval_fn)
      Map.has_key?(known_pairs, {b, a}) and known_pairs[{b, a}] != :failed ->
        not PredProg.eval(known_pairs[{b, a}], env_mod.state_to_list(state), eval_fn)
      true -> false
    end
  end

  @doc "Rolls out a trajectory and accumulates the penalty."
  def penalty_rollout(_state, _action, _p, _action_a, _action_b, _env_mod, _eval_fn, 0, _known_pairs), do: 0
  def penalty_rollout(state, action, p, action_a, action_b, env_mod, eval_fn, k, known_pairs) do
    if env_mod.terminal?(state) do
      env_mod.penalty(state)
    else
      {s_prime, accumulated_penalty} = apply_substeps(state, action, env_mod, 5, 0)
      next_action = tournament_action(s_prime, p, action_a, action_b, known_pairs, env_mod, eval_fn)
      accumulated_penalty + penalty_rollout(s_prime, next_action, p, action_a, action_b, env_mod, eval_fn, k - 1, known_pairs)
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

  @doc "Returns true if action_a is strictly better, false if action_b is strictly better, or :tie."
  def oracle_predict(p, s, action_a, action_b, env_mod, eval_fn, known_pairs) do
    k = div(env_mod.oracle_horizon(), 5)
    penalty_a = penalty_rollout(s, action_a, p, action_a, action_b, env_mod, eval_fn, k, known_pairs)
    penalty_b = penalty_rollout(s, action_b, p, action_a, action_b, env_mod, eval_fn, k, known_pairs)

    cond do
      penalty_a >= env_mod.crash_penalty() and penalty_b >= env_mod.crash_penalty() -> :tie
      penalty_a == penalty_b -> :tie
      penalty_a < penalty_b -> true
      true -> false
    end
  end

  @doc "Computes the sum score across all starting states for a candidate predicate."
  def sum_score(p, action_a, action_b, env_mod, eval_fn, known_pairs) do
    h = div(env_mod.score_horizon(), 5)

    Enum.reduce(env_mod.starts(), 0, fn s0, acc ->
      acc + penalty_score(s0, p, action_a, action_b, env_mod, eval_fn, h, known_pairs)
    end)
  end

  defp penalty_score(_state, _p, _action_a, _action_b, _env_mod, _eval_fn, 0, _known_pairs), do: 0
  defp penalty_score(state, p, action_a, action_b, env_mod, eval_fn, k, known_pairs) do
    if env_mod.terminal?(state) do
      0
    else
      action = tournament_action(state, p, action_a, action_b, known_pairs, env_mod, eval_fn)
      {s_prime, _} = apply_substeps(state, action, env_mod, 5, 0)

      step_score = env_mod.max_penalty() - env_mod.penalty(state)
      step_score + penalty_score(s_prime, p, action_a, action_b, env_mod, eval_fn, k - 1, known_pairs)
    end
  end

  @doc "Collects a trajectory under the current pairwise policy."
  def collect(_state, _p, _action_a, _action_b, _env_mod, _eval_fn, 0, _known_pairs), do: []
  def collect(state, p, action_a, action_b, env_mod, eval_fn, n, known_pairs) do
    if env_mod.terminal?(state) do
      []
    else
      action = tournament_action(state, p, action_a, action_b, known_pairs, env_mod, eval_fn)
      {s_prime, _} = apply_substeps(state, action, env_mod, 5, 0)
      [state | collect(s_prime, p, action_a, action_b, env_mod, eval_fn, n - 1, known_pairs)]
    end
  end
end
