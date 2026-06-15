defmodule Synthex.Verifier do
  @moduledoc """
  Pluggable counterexample source for one CEGAR step.

  CEGIS power lives in the verifier — the component that finds the states
  a candidate policy handles badly. Historically Synthex used a *passive*
  verifier: on-policy `collect_states` for the feature domain plus a
  deterministic block of random `seeds_for` for scoring. This behaviour
  makes that step a strategy so it can be swapped for an *active*
  adversary without touching the synthesizer or the commit-gate.

  Implementations return the material a CEGAR step refines against:

    * `:states`         — observed states feeding `build_features/2`
    * `:seeds`          — the seeds used for same-seed candidate scoring
    * `:counterexamples`— `[%{...}]` diagnostics (empty for `:random`)

  Selection is by `ctx.verifier` (`:random` default, `:ga_qd` opt-in).
  See `docs/ga-counterexample-verifier.md` in synthex-hub.
  """

  @type supply_result :: %{
          states: [list()],
          seeds: [non_neg_integer()],
          counterexamples: [map()]
        }

  @callback supply(preds :: list(), ctx :: map(), round :: pos_integer()) :: supply_result()

  @doc """
  Dispatch to the configured verifier strategy. Defaults to
  `Synthex.Verifier.RandomSeeds`, which is behaviour-identical to the
  pre-existing inline `collect_states` + `seeds_for` pair.
  """
  @spec supply(list(), map(), pos_integer()) :: supply_result()
  def supply(preds, ctx, round) do
    impl(Map.get(ctx, :verifier, :random)).supply(preds, ctx, round)
  end

  defp impl(:ga_qd), do: Synthex.Verifier.GAQD
  defp impl(_), do: Synthex.Verifier.RandomSeeds
end
