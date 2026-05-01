defmodule Synthex do
  @moduledoc """
  Synthex: a concurrent synthesis engine for Coinductive Symmetric
  Homomorphism Reinforcement Learning (CSHRL).

  Discovers formally verifiable policies expressed as Boolean predicate
  programs (`PredProg`) by aggressively searching the candidate space
  across CPU cores using Elixir's `Flow` and `GenStage`.
  """
end

# Backward-compatible aliases so internal modules can reference
# Synthex.PredProg, Synthex.CEGIS, etc. without the Core prefix.
defmodule Synthex.PredProg do
  @moduledoc false
  defdelegate eval(prog, state, eval_feat_fn), to: Synthex.Core.PredProg
  @type t :: Synthex.Core.PredProg.t()
end

defmodule Synthex.CEGIS do
  @moduledoc false
  defdelegate enumerate(features, depth), to: Synthex.Core.CEGIS
  defdelegate refine(version_space, observations, eval_feat_fn), to: Synthex.Core.CEGIS
end

defmodule Synthex.ContinuousFeatures do
  @moduledoc false
  defdelegate generate(trajectory, num_dims, max_coeff), to: Synthex.Core.ContinuousFeatures
  defdelegate eval_feature(feature, state), to: Synthex.Core.ContinuousFeatures
end

defmodule Synthex.Environment do
  @moduledoc false
  # Re-export the behaviour callbacks so `@behaviour Synthex.Environment` still works
  @callback step(state :: any(), action :: any()) :: any()
  @callback terminal?(state :: any()) :: boolean()
  @callback penalty(state :: any()) :: integer()
  @callback crash_penalty() :: integer()
  @callback max_penalty() :: integer()
  @callback starts() :: [any()]
  @callback actions() :: [any()]
  @callback state_to_list(state :: any()) :: [integer()]
  @callback oracle_horizon() :: integer()
  @callback score_horizon() :: integer()
end
