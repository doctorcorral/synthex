defmodule Synthex.Pure.Permutation.Oracle do
  @moduledoc """
  An oracle designed for Permutation Policies.
  Instead of answering pairwise questions, it evaluates the full policy 
  via rollouts and returns a cumulative penalty score.
  """

  alias Synthex.Pure.Permutation.Policy

  @doc """
  Rolls out a trajectory using the given Permutation Policy and returns the total penalty.
  """
  def penalty_rollout(_state, _policy, _env_mod, _eval_fn, 0), do: 0
  def penalty_rollout(state, policy, env_mod, eval_fn, k) do
    if env_mod.terminal?(state) do
      env_mod.penalty(state)
    else
      action = Policy.best_action(policy, env_mod.state_to_list(state), eval_fn)
      s_prime = env_mod.step(state, action)
      env_mod.penalty(state) + penalty_rollout(s_prime, policy, env_mod, eval_fn, k - 1)
    end
  end

  @doc """
  Computes the sum score across all starting states for a given policy.
  Lower is better.
  """
  def sum_penalty(policy, env_mod, eval_fn) do
    h = env_mod.score_horizon()
    
    Enum.reduce(env_mod.starts(), 0, fn s0, acc -> 
      acc + penalty_rollout(s0, policy, env_mod, eval_fn, h)
    end)
  end
end
