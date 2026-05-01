defmodule Synthex.Core.PredProgTest do
  use ExUnit.Case, async: true

  alias Synthex.Core.PredProg

  defp eval_fn do
    fn
      {:axis, dim, threshold}, state -> Enum.at(state, dim) < threshold
      {:diag, i, j, c}, state -> c * Enum.at(state, i) + Enum.at(state, j) < 0
    end
  end

  test "truep evaluates to true" do
    assert PredProg.eval(:truep, [1, 2, 3], eval_fn())
  end

  test "falsep evaluates to false" do
    refute PredProg.eval(:falsep, [1, 2, 3], eval_fn())
  end

  test "axis feature below threshold" do
    assert PredProg.eval({:feat, {:axis, 0, 5}}, [3, 10], eval_fn())
  end

  test "axis feature above threshold" do
    refute PredProg.eval({:feat, {:axis, 0, 5}}, [7, 10], eval_fn())
  end

  test "not negates" do
    refute PredProg.eval({:not, :truep}, [1], eval_fn())
    assert PredProg.eval({:not, :falsep}, [1], eval_fn())
  end

  test "and requires both" do
    p = {:and, {:feat, {:axis, 0, 5}}, {:feat, {:axis, 1, 10}}}
    assert PredProg.eval(p, [3, 8], eval_fn())
    refute PredProg.eval(p, [3, 12], eval_fn())
    refute PredProg.eval(p, [7, 8], eval_fn())
  end

  test "or requires either" do
    p = {:or, {:feat, {:axis, 0, 5}}, {:feat, {:axis, 1, 10}}}
    assert PredProg.eval(p, [3, 12], eval_fn())
    assert PredProg.eval(p, [7, 8], eval_fn())
    refute PredProg.eval(p, [7, 12], eval_fn())
  end

  test "diagonal feature" do
    p = {:feat, {:diag, 0, 1, 2}}
    assert PredProg.eval(p, [-3, 4], eval_fn())
    refute PredProg.eval(p, [3, 4], eval_fn())
  end
end
