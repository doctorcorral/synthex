defmodule Synthex.Verifier.RandomSeeds do
  @moduledoc """
  Default verifier — the historical, passive behaviour.

  The feature domain is the on-policy state distribution (`collect_states`)
  and the scoring seeds are the deterministic `seeds_for` block. This is a
  verbatim wrapper of what `Synthex.Gym.Mujoco.solve/2` and the hub
  controller did inline, so selecting `verifier: :random` (the default) is
  bit-for-bit equivalent to the pre-verifier code path.
  """

  @behaviour Synthex.Verifier

  alias Synthex.Gym.Mujoco

  @impl true
  def supply(preds, ctx, round) do
    {states, _n_landings, snapshots} = Mujoco.collect_states(preds, ctx)
    seeds = Mujoco.seeds_for(round, 1, ctx)
    base = %{states: states, seeds: seeds, counterexamples: []}

    if snapshots != [] do
      Map.put(base, :succ_snapshots, snapshots)
    else
      base
    end
  end
end
