defmodule Synthex.Core.CEGISTest do
  use ExUnit.Case, async: true

  alias Synthex.Core.CEGIS

  test "enumerate depth 0 includes truep, falsep, and features" do
    features = [{:axis, 0, 5}, {:axis, 1, 3}]
    result = CEGIS.enumerate(features, 0)

    assert :truep in result
    assert :falsep in result
    assert {:feat, {:axis, 0, 5}} in result
    assert {:feat, {:axis, 1, 3}} in result
    assert length(result) == 4
  end

  test "enumerate depth 1 expands with boolean connectives" do
    features = [{:axis, 0, 5}]
    result = CEGIS.enumerate(features, 1)

    assert {:and, :truep, {:feat, {:axis, 0, 5}}} in result
    assert {:or, :falsep, {:feat, {:axis, 0, 5}}} in result
    assert {:not, {:feat, {:axis, 0, 5}}} in result
    assert length(result) > length(CEGIS.enumerate(features, 0))
  end

  test "refine filters by consistency" do
    eval_fn = fn {:axis, dim, t}, state -> Enum.at(state, dim) < t end

    version_space = [
      {:feat, {:axis, 0, 5}},
      {:feat, {:axis, 0, 2}},
      :truep,
      :falsep
    ]

    observations = [
      {[3], true},
      {[7], false}
    ]

    result = CEGIS.refine(version_space, observations, eval_fn)

    assert {:feat, {:axis, 0, 5}} in result
    refute {:feat, {:axis, 0, 2}} in result
    refute :truep in result
    refute :falsep in result
  end
end
