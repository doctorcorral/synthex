defmodule Synthex.Core.Environment do
  @moduledoc """
  A formal behaviour defining the physics and constraints of a continuous state
  environment for Pairwise CSHRL Synthesis.
  """

  @doc "Applies the chosen action to the state and returns the next state."
  @callback step(state :: any(), action :: any()) :: any()

  @doc "Returns true if the episode should end (e.g., bounds exceeded, crash)."
  @callback terminal?(state :: any()) :: boolean()

  @doc "Returns the immediate scalar penalty for being in the given state."
  @callback penalty(state :: any()) :: integer()

  @doc "Returns the massive penalty applied if terminal? is true due to a failure state."
  @callback crash_penalty() :: integer()

  @doc "Returns the maximum allowed penalty per step (for converting penalties to scores)."
  @callback max_penalty() :: integer()

  @doc "A list of diverse starting states used to seed the StateCEGAR anchors and evaluate score."
  @callback starts() :: [any()]

  @doc "A list of all possible valid actions in the environment."
  @callback actions() :: [any()]

  @doc "Converts a tuple state into a flat list of integers for structural feature generation."
  @callback state_to_list(state :: any()) :: [integer()]

  @doc "The lookahead horizon K for the oracle."
  @callback oracle_horizon() :: integer()

  @doc "The rollout horizon H for evaluating sum_score."
  @callback score_horizon() :: integer()
end
