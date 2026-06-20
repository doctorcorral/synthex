defmodule Synthex.Gym.SuccessorScorerTest do
  use ExUnit.Case, async: true

  alias Synthex.Gym.SuccessorScorer
  alias Synthex.Gym.Oracle, as: GymOracle

  test "score_successor_local fires where advantage is positive" do
    obs = [[0.0], [1.0], [2.0]]
    snapshots = Enum.map(obs, fn o -> %{"obs" => o} end)
    advantages = [0.0, 2.5, -1.0]

    candidates = [:falsep, :truep, {:feat, ["axis", 0, 0.5]}]

    {scored, baseline} =
      SuccessorScorer.score_bit_candidates(
        candidates,
        List.duplicate(:falsep, 3),
        0,
        [0],
        %{
          succ_snapshots: snapshots,
          successor_lookahead: 10,
          cfg: %{gym_name: "Walker2d-v5", num_dims: 1, n_action_dims: 1, action_range: {-1, 1}},
          bits_per_dim: 1,
          max_steps: 100,
          scorer: fn %{"cmd" => "successor_advantages"} ->
            {:ok, %{"advantages" => advantages}}
          end
        }
      )

    assert baseline == 0.0
    by_idx = Map.new(scored, fn {i, r, _} -> {i, r} end)
    assert by_idx[0] == 0.0
    assert by_idx[1] == 1.5
    assert by_idx[2] == 0.0
  end

  test "eval_pred used by successor local scorer matches oracle" do
    obs = [0.3, -0.2, 0.9]
    pred = {:feat, ["axis", 0, 0.5]}
    assert GymOracle.eval_pred(pred, obs)
  end
end
