defmodule Synthex.Core.ContinuousFeatures do
  @moduledoc """
  Generates geometric features from continuous trajectories.
  Mirrors `CSHRL.Synthesis.ContinuousFeatures` from Agda.
  """

  @doc """
  Generates axis-aligned and diagonal boundary features.

  ## Parameters
  - `trajectory`: A list of states (where a state is a list/tuple of numbers)
  - `num_dims`: The dimensionality of the state space.
  - `max_coeff`: The maximum integer coefficient for diagonal boundaries.
  """
  def generate(trajectory, num_dims, max_coeff) do
    thresholds =
      Enum.reduce(0..(num_dims - 1), %{}, fn dim_idx, acc ->
        vals = Enum.map(trajectory, fn state -> Enum.at(state, dim_idx) end)
        Map.put(acc, dim_idx, Enum.uniq([0 | vals]))
      end)

    axis_feats = generate_axis_features(thresholds)
    diag_feats = generate_diag_features(num_dims, max_coeff)

    axis_feats ++ diag_feats
  end

  defp generate_axis_features(thresholds_map) do
    for {dim_idx, ts} <- thresholds_map, t <- ts do
      {:axis, dim_idx, t}
    end
  end

  defp generate_diag_features(num_dims, max_coeff) do
    coeffs = for c <- 1..max_coeff, val <- [c, -c], do: val

    for i <- 0..(num_dims - 1),
        j <- 0..(num_dims - 1),
        i != j,
        c <- coeffs do
      {:diag, i, j, c}
    end
  end

  @doc """
  The standard evaluation function for continuous features.
  """
  def eval_feature({:axis, dim, threshold}, state) do
    Enum.at(state, dim) < threshold
  end

  def eval_feature({:diag, i, j, c}, state) do
    (c * Enum.at(state, i)) + Enum.at(state, j) < 0
  end
end
