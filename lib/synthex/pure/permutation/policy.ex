defmodule Synthex.Pure.Permutation.Policy do
  @moduledoc """
  Defines the Permutation Policy structure and evaluation logic.
  A policy consists of a base ranking of actions and a list of conditional swaps.
  """

  @type base_ranking :: [atom()]
  @type swap :: {integer(), integer()}
  @type cond_swap :: {Synthex.Core.PredProg.t(), swap()}
  
  @type t :: %__MODULE__{
    base_ranking: base_ranking(),
    swaps: [cond_swap()]
  }

  defstruct [:base_ranking, :swaps]

  @doc """
  Applies a single swap (i, j) to a list.
  If indices are out of bounds, returns the list unmodified.
  """
  def apply_swap(list, {i, j}) do
    len = length(list)
    if i >= 0 and j >= 0 and i < len and j < len do
      val_i = Enum.at(list, i)
      val_j = Enum.at(list, j)
      
      list
      |> List.replace_at(i, val_j)
      |> List.replace_at(j, val_i)
    else
      list
    end
  end

  @doc """
  Evaluates the Permutation Policy for a given state.
  Returns the final ordered list of actions (index 0 is best).
  """
  def evaluate(%__MODULE__{base_ranking: base, swaps: swaps}, state, eval_fn) do
    Enum.reduce(swaps, base, fn {cond_prog, swap_op}, current_ranking ->
      if Synthex.Core.PredProg.eval(cond_prog, state, eval_fn) do
        apply_swap(current_ranking, swap_op)
      else
        current_ranking
      end
    end)
  end

  @doc """
  Returns the single best action (the head of the evaluated list) for a state.
  """
  def best_action(policy, state, eval_fn) do
    hd(evaluate(policy, state, eval_fn))
  end
end
