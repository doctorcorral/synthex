defmodule Synthex.Core.PredProg do
  @moduledoc """
  The Boolean Predicate Domain Specific Language (DSL).
  This mirrors the Agda `PredProg` data type for synthesis.
  """

  @type t ::
          :truep
          | :falsep
          | {:feat, any()}
          | {:not, t()}
          | {:and, t(), t()}
          | {:or, t(), t()}

  @doc """
  Evaluates a predicate program against a state using a feature evaluation function.
  """
  @spec eval(t(), state :: any(), (any(), any() -> boolean())) :: boolean()
  def eval(:truep, _state, _eval_feat_fn), do: true
  def eval(:falsep, _state, _eval_feat_fn), do: false

  def eval({:feat, feature}, state, eval_feat_fn) do
    eval_feat_fn.(feature, state)
  end

  def eval({:not, p}, state, eval_feat_fn) do
    not eval(p, state, eval_feat_fn)
  end

  def eval({:and, p1, p2}, state, eval_feat_fn) do
    eval(p1, state, eval_feat_fn) and eval(p2, state, eval_feat_fn)
  end

  def eval({:or, p1, p2}, state, eval_feat_fn) do
    eval(p1, state, eval_feat_fn) or eval(p2, state, eval_feat_fn)
  end
end
