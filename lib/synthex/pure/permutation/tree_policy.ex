defmodule Synthex.Pure.Permutation.TreePolicy do
  @moduledoc """
  Defines the Permutation Tree (RankTree) structure and evaluation logic.
  Instead of a flat list of conditional swaps, this forms a decision tree 
  where each leaf node is a pure deterministic permutation (a Base Ranking),
  and each inner node is a Depth-0 PredProg branch.
  """

  @type ranking :: [atom()]
  
  # A Leaf just returns a static ranking of actions
  @type leaf :: {:leaf, ranking()}
  
  # A Branch evaluates a predicate. If true, takes the true_branch, else false_branch.
  @type branch :: {:branch, Synthex.Core.PredProg.t(), t(), t()}

  @type t :: leaf() | branch()

  @doc """
  Evaluates the PermTree for a given state to get the final ordered list of actions.
  """
  def evaluate({:leaf, ranking}, _state, _eval_fn), do: ranking

  def evaluate({:branch, cond_prog, true_branch, false_branch}, state, eval_fn) do
    if Synthex.Core.PredProg.eval(cond_prog, state, eval_fn) do
      evaluate(true_branch, state, eval_fn)
    else
      evaluate(false_branch, state, eval_fn)
    end
  end

  @doc """
  Returns the single best action (the head of the evaluated list) for a state.
  """
  def best_action(tree, state, eval_fn) do
    hd(evaluate(tree, state, eval_fn))
  end
end
