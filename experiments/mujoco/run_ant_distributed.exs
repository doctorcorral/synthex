# Ant — distributed CSHRL synthesis via Synthex Hub.
#
# Friends donate compute by running:
#   curl -fsSL https://synthex.fit/install | sh
#
# Then on YOUR (operator) machine, with SYNTHEX_HUB_TOKEN exported:
#
#   cd /Users/rcc/Research/synthex
#   mix run experiments/mujoco/run_ant_distributed.exs
#
# Local prerequisites (master only): gymnasium + mujoco for state
# collection / validation. The heavy candidate-scoring work is
# farmed out to the hub.

require Logger

# Sanity check: any workers connected?
client = Synthex.Hub.Client.new()
case Synthex.Hub.Client.public_status(client) do
  {:ok, %{"active_workers" => 0}} ->
    IO.puts("\n  WARNING: 0 active workers connected at #{client.base_url}")
    IO.puts("  Tell collaborators to run: curl -fsSL https://synthex.fit/install | sh")
    IO.puts("  (continuing anyway — chunks will queue and run when workers connect)\n")

  {:ok, %{"active_workers" => n, "total_cores" => c}} ->
    IO.puts("\n  Cluster: #{n} worker(s), #{c} core(s) ready.\n")

  {:error, reason} ->
    IO.puts("\n  Could not reach hub: #{inspect(reason)}\n")
end

Synthex.Gym.Mujoco.solve(:ant,
  # Distributed execution
  executor: :hub,
  # hub_url + hub_token come from SYNTHEX_HUB_URL / SYNTHEX_HUB_TOKEN
  # env vars (or hub_url:/hub_token: kw opts here).
  hub_chunk_size: 100,
  hub_poll_interval_ms: 5_000,
  hub_batch_name: "ant-#{DateTime.utc_now() |> DateTime.to_iso8601()}",

  # Synthesis hyper-parameters
  bits_per_dim: 3,
  depth: 1,
  max_coeff: 5,

  # All five feature classes including tridiag.
  feature_types: [:axis, :diag, :sq_diag, :prod, :tridiag],
  # Ant has 105 obs dims; full tridiag would be ~10M features. Cap
  # the coefficient bound and restrict to qpos+qvel (first 27 dims)
  # to keep this tractable on a 32-core swarm.
  tridiag_max_coeff: 2,
  tridiag_dims: 0..26,

  n_episodes: 30,
  top_k: 24,
  max_iters: 5,
  cegar_rounds: 3,
  max_steps: 1000
)
