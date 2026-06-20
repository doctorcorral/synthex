defmodule Synthex.Gym.SuccessorScorer do
  @moduledoc """
  CoinductiveHomomorphism fitness for continuous bit-policy synthesis.

  Phase 1 scores every candidate by successor-value dominance:

      A(s) = V(next(s, a1)) - V(next(s, a0))

  where `a0`/`a1` differ only in `target_bit`. The value-maximizing
  predicate maximizes `Sum_s A(s) * p(s)`. Candidate scoring is pure
  Elixir once the worker returns the per-snapshot advantage vector.

  Phase 2 (episode commit verification) runs in `Mujoco.do_optimize_bit/5`
  after a successor winner is chosen, so the commit gate still sees a
  same-seed episode baseline/improvement pair.
  """

  alias Synthex.Gym.Mujoco
  alias Synthex.Gym.Oracle, as: GymOracle

  @doc """
  Successor Phase 1 scoring for one bit.

  Returns `{scored, baseline}` in the same shape as episode scoring, but
  both use the **successor** objective: `scored` ranks candidates by
  `Sum_s A(s)*p(s)` and `baseline` is the current bit predicate's
  successor score. Episode verification for the commit gate happens in
  `Mujoco.maybe_episode_verify_commit/5`.
  """
  @spec score_bit_candidates(
          [Synthex.Core.PredProg.predicate()],
          [Synthex.Core.PredProg.predicate()],
          non_neg_integer(),
          [non_neg_integer()],
          map()
        ) :: {[{non_neg_integer(), number(), number()}], float()}
  def score_bit_candidates(candidates, preds, target_bit, seeds, ctx) do
    snapshots = Map.get(ctx, :succ_snapshots) || []

    if snapshots == [] do
      raise "successor fitness requires :succ_snapshots in ctx " <>
              "(collect_states with want_sim_state on a MuJoCo env)"
    end

    advantages = fetch_advantages(preds, target_bit, snapshots, ctx)
    obs_list = Enum.map(snapshots, &snapshot_obs/1)
    succ_scored = score_successor_local(candidates, advantages, obs_list)

    current_pred = Enum.at(preds, target_bit)

    succ_baseline =
      case score_successor_local([current_pred], advantages, obs_list) do
        [{_, b, _}] -> b
        _ -> 0.0
      end

    {succ_scored, succ_baseline}
  end

  @doc """
  Episode-return score for a single predicate at `target_bit`, plus the
  current vector's episode baseline on the same seeds.
  """
  @spec episode_score_winner(
          Synthex.Core.PredProg.predicate(),
          [Synthex.Core.PredProg.predicate()],
          non_neg_integer(),
          [non_neg_integer()],
          map()
        ) :: {float(), float()}
  def episode_score_winner(winner, preds, target_bit, seeds, ctx) do
    {scored, baseline} =
      Mujoco.score_bit_candidates_episode([winner], preds, target_bit, seeds, ctx)

    case scored do
      [{_idx, reward, _}] when is_number(reward) -> {reward, baseline}
      _ -> {baseline, baseline}
    end
  end

  defp score_successor_local(candidates, advantages, obs_list) do
    candidates
    |> Enum.with_index()
    |> Enum.map(fn {pred, idx} ->
      total =
        obs_list
        |> Enum.with_index()
        |> Enum.reduce(0.0, fn {obs, i}, acc ->
          if GymOracle.eval_pred(pred, obs) do
            acc + Enum.at(advantages, i)
          else
            acc
          end
        end)

      {idx, total, 0}
    end)
  end

  defp fetch_advantages(preds, target_bit, snapshots, ctx) do
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)

    request = %{
      "cmd" => "successor_advantages",
      "env_name" => ctx.cfg.gym_name,
      "env_spec" => Mujoco.env_spec(ctx),
      "bits_per_dim" => ctx.bits_per_dim,
      "bit_predicates" => serialized_preds,
      "target_bit" => target_bit,
      "snapshots" => snapshots,
      "lookahead" => Map.get(ctx, :successor_lookahead, 40),
      "max_steps" => ctx.max_steps
    }

    result = Mujoco.invoke_scorer!(request, ctx)
    result["advantages"] || []
  end

  defp snapshot_obs(%{"obs" => obs}) when is_list(obs), do: obs
  defp snapshot_obs(%{obs: obs}) when is_list(obs), do: obs
  defp snapshot_obs(obs) when is_list(obs), do: obs
end
