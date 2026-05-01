defmodule Synthex.Envs.MountainCar do
  @behaviour Synthex.Environment
  
  @s 100_000_000

  def div_int(a, b), do: div(a, b)
  def f_mul(a, b), do: div_int(a * b, @s)

  def cos7(y) do
    y2 = f_mul(y, y)
    y4 = f_mul(y2, y2)
    y6 = f_mul(y4, y2)
    y8 = f_mul(y6, y2)
    y10 = f_mul(y8, y2)
    y12 = f_mul(y10, y2)
    @s - div_int(y2, 2) + div_int(y4, 24) - div_int(y6, 720) + div_int(y8, 40320) - div_int(y10, 3628800) + div_int(y12, 479001600)
  end

  def fclip(lo, hi, x) do
    cond do
      x < lo -> lo
      x > hi -> hi
      true -> x
    end
  end

  def wall(x_prime, v_prime) do
    if x_prime <= -120_000_000 do
      if v_prime < 0, do: 0, else: v_prime
    else
      v_prime
    end
  end

  def act_f(:push_left), do: -100_000
  def act_f(:no_action), do: 0
  def act_f(:push_right), do: 100_000

  def sub_step({x, v}, a) do
    grav = f_mul(cos7(f_mul(300_000_000, x)), -250_000)
    v_prime = fclip(-7_000_000, 7_000_000, v + act_f(a) + grav)
    x_prime = fclip(-120_000_000, 60_000_000, x + v_prime)
    {x_prime, wall(x_prime, v_prime)}
  end

  def multi_step(0, st, _a), do: st
  def multi_step(n, st, a), do: multi_step(n - 1, sub_step(st, a), a)

  @impl true
  def step(state, a) do
    multi_step(25, state, a)
  end

  @impl true
  def terminal?({x, _}) do
    x >= 50_000_000
  end

  @impl true
  def penalty(_) do
    1 # 1 step penalty to encourage finding the goal quickly
  end

  @impl true
  def crash_penalty(), do: 0 # No crash in MC

  @impl true
  def max_penalty(), do: 200 # Max steps

  @impl true
  def actions(), do: [:push_left, :no_action, :push_right]

  @impl true
  def starts do
    [
      {-50_000_000, 0},
      {-25_000_000, 0},
      {-75_000_000, 0}
    ]
  end

  @impl true
  def state_to_list({x, v}), do: [x, v]

  @impl true
  def oracle_horizon(), do: 200

  @impl true
  def score_horizon(), do: 200
end
