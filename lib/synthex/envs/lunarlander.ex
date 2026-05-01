defmodule Synthex.Envs.LunarLander do
  @behaviour Synthex.Environment

  @s 100_000_000
  def div_int(a, b), do: div(a, b)
  def f_mul(a, b), do: div_int(a * b, @s)

  def sin7(y) do
    y2 = f_mul(y, y)
    y3 = f_mul(y2, y)
    y5 = f_mul(y3, y2)
    y7 = f_mul(y5, y2)
    y9 = f_mul(y7, y2)
    y11 = f_mul(y9, y2)
    y13 = f_mul(y11, y2)
    y - div_int(y3, 6) + div_int(y5, 120) - div_int(y7, 5040) + div_int(y9, 362880) - div_int(y11, 39916800) + div_int(y13, 6227020800)
  end

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

  @g_acc 300_000_000
  @m_acc 800_000_000
  @s_lat 50_000_000
  @s_rot 200_000_000
  @dt 2_000_000
  @v_max 500_000_000
  @o_max 500_000_000
  @y_max 300_000_000

  def act_ax(:fire_main, theta), do: -f_mul(@m_acc, sin7(theta))
  def act_ax(:fire_left, _), do: -@s_lat
  def act_ax(:fire_right, _), do: @s_lat
  def act_ax(:do_nothing, _), do: 0

  def act_ay_eng(:fire_main, theta), do: f_mul(@m_acc, cos7(theta))
  def act_ay_eng(_, _), do: 0

  def act_alpha(:fire_left), do: @s_rot
  def act_alpha(:fire_right), do: -@s_rot
  def act_alpha(_), do: 0

  @impl true
  def step({x, y, vx, vy, theta, omega}, a) do
    ax = act_ax(a, theta)
    aye = act_ay_eng(a, theta)
    alpha = act_alpha(a)
    
    vx_prime = fclip(-@v_max, @v_max, vx + f_mul(ax, @dt))
    vy_prime = fclip(-@v_max, @v_max, vy + f_mul(-@g_acc + aye, @dt))
    x_prime = x + f_mul(vx_prime, @dt)
    y_prime = fclip(0, @y_max, y + f_mul(vy_prime, @dt))
    omega_prime = fclip(-@o_max, @o_max, omega + f_mul(alpha, @dt))
    theta_prime = theta + f_mul(omega_prime, @dt)
    
    {x_prime, y_prime, vx_prime, vy_prime, theta_prime, omega_prime}
  end

  @impl true
  def terminal?({x, y, _vx, _vy, theta, _omega}) do
    y <= 0 or abs(x) > 100_000_000 or abs(theta) > 150_000_000
  end

  @impl true
  def penalty({x, y, vx, vy, theta, _omega}) do
    xr = div_int(abs(x), 100_000)
    yr = div_int(abs(y), 100_000)
    vxr = div_int(abs(vx), 100_000)
    vyr = div_int(abs(vy), 100_000)
    tr = div_int(abs(theta), 100_000)
    xr + yr + vxr + vyr + tr
  end

  @impl true
  def crash_penalty(), do: 100_000_000

  @impl true
  def max_penalty(), do: 10_000

  @impl true
  def actions(), do: [:fire_main, :fire_left, :fire_right, :do_nothing]

  @impl true
  def starts do
    [
      {0, 150_000_000, 0, 0, 0, 0},
      {50_000_000, 150_000_000, 0, 0, 0, 0},
      {-50_000_000, 150_000_000, 0, 0, 0, 0},
      {0, 150_000_000, 0, 0, 20_000_000, 0},
      {0, 150_000_000, 0, 0, -20_000_000, 0},
      {0, 150_000_000, 50_000_000, 0, 0, 0},
      {0, 150_000_000, -50_000_000, 0, 0, 0},
      {0, 150_000_000, 0, 0, 0, 20_000_000},
      {0, 150_000_000, 0, 0, 0, -20_000_000},
      {50_000_000, 100_000_000, 50_000_000, -50_000_000, 50_000_000, 20_000_000},
      {-50_000_000, 100_000_000, -50_000_000, -50_000_000, -50_000_000, -20_000_000}
    ]
  end

  @impl true
  def state_to_list({x, y, vx, vy, theta, omega}), do: [x, y, vx, vy, theta, omega]

  @impl true
  def oracle_horizon(), do: 150

  @impl true
  def score_horizon(), do: 200
end
