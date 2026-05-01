defmodule Synthex.Gym.OracleTest do
  use ExUnit.Case, async: true

  alias Synthex.Gym.Oracle

  test "action_map returns correct maps for all registered envs" do
    assert Oracle.action_map(:lunarlander) == %{do_nothing: 0, fire_left: 1, fire_main: 2, fire_right: 3}
    assert Oracle.action_map(:tetris) == %{left: 0, right: 1, rotate: 2, down: 3, noop: 4}
    assert Oracle.action_map(:pendulum) == %{torque_neg: 0, no_torque: 1, torque_pos: 2}
    assert Oracle.action_map(:cartpole) == %{left: 0, right: 1}
  end

  test "num_dims returns correct dimensionality" do
    assert Oracle.num_dims(:lunarlander) == 6
    assert Oracle.num_dims(:pendulum) == 3
    assert Oracle.num_dims(:tetris) == 8
    assert Oracle.num_dims(:cartpole) == 4
  end

  test "serialize_pred handles all AST forms" do
    assert Oracle.serialize_pred(:truep) == "truep"
    assert Oracle.serialize_pred(:falsep) == "falsep"
    assert Oracle.serialize_pred({:feat, ["axis", 0, 5]}) == ["feat", ["axis", 0, 5]]
    assert Oracle.serialize_pred({:not, :truep}) == ["not", "truep"]
    assert Oracle.serialize_pred({:and, :truep, :falsep}) == ["and", "truep", "falsep"]
  end

  test "serialize_chain produces correct format" do
    chain = [{:truep, :fire_left}, {:falsep, :fire_main}]
    result = Oracle.serialize_chain(chain, :lunarlander)
    assert result == [["truep", 1], ["falsep", 2]]
  end

  test "eval_pred handles axis features" do
    assert Oracle.eval_pred({:feat, ["axis", 0, 5]}, [3, 10])
    refute Oracle.eval_pred({:feat, ["axis", 0, 5]}, [7, 10])
  end

  test "eval_chain selects correct action" do
    chain = [
      {{:feat, ["axis", 0, 5]}, :fire_left},
      {{:feat, ["axis", 1, 10]}, :fire_right}
    ]

    assert Oracle.eval_chain(chain, :do_nothing, [3, 8]) == :fire_left
    assert Oracle.eval_chain(chain, :do_nothing, [7, 8]) == :fire_right
    assert Oracle.eval_chain(chain, :do_nothing, [7, 12]) == :do_nothing
  end

  test "format_pred produces readable output" do
    assert Oracle.format_pred(:truep, :lunarlander) == "true"
    assert Oracle.format_pred({:feat, ["axis", 0, 5]}, :lunarlander) == "x<5"
    assert Oracle.format_pred({:feat, ["diag", 0, 1, 3]}, :lunarlander) == "3·x+y<0"
  end

  test "reverse_action_map inverts correctly" do
    rev = Oracle.reverse_action_map(:lunarlander)
    assert rev[0] == :do_nothing
    assert rev[1] == :fire_left
    assert rev[2] == :fire_main
    assert rev[3] == :fire_right
  end
end
