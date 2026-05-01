defmodule Synthex.Envs.LunarLanderGym do
  @moduledoc """
  LunarLander environment calibrated to Gymnasium LunarLander-v3 physics.

  Measured accelerations (observation-space units):
    Gravity:      -1.284 m/s²
    Main thrust:   2.336 m/s² (upward, net +1.052 with gravity)
    Side lateral:  0.50  m/s²
    Side angular:  2.00  rad/s²
    dt = 0.02s (50 FPS)
  """

  @behaviour Synthex.Environment

  @s 100_000_000
  def f_mul(a, b), do: div(a * b, @s)

  def fclip(lo, hi, x) do
    cond do
      x < lo -> lo
      x > hi -> hi
      true -> x
    end
  end

  @g_acc  128_400_000   # 1.284 m/s² (calibrated from Gymnasium)
  @m_acc  233_600_000   # 2.336 m/s² (main engine, calibrated)
  @s_lat   50_000_000   # 0.50  m/s² (lateral, matches)
  @s_rot  200_000_000   # 2.00  rad/s² (angular, matches)
  @dt       2_000_000   # 0.02s
  @v_max  500_000_000   # 5.0 (safety clamp)
  @o_max  500_000_000   # 5.0
  @y_max  300_000_000   # 3.0

  def sin_approx(y) do
    y2 = f_mul(y, y)
    y3 = f_mul(y2, y)
    y5 = f_mul(y3, y2)
    y - div(y3, 6) + div(y5, 120)
  end

  def cos_approx(y) do
    y2 = f_mul(y, y)
    y4 = f_mul(y2, y2)
    @s - div(y2, 2) + div(y4, 24)
  end

  def act_ax(:fire_main, theta), do: -f_mul(@m_acc, sin_approx(theta))
  def act_ax(:fire_left, _), do: -@s_lat
  def act_ax(:fire_right, _), do: @s_lat
  def act_ax(:do_nothing, _), do: 0

  def act_ay_eng(:fire_main, theta), do: f_mul(@m_acc, cos_approx(theta))
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
  def penalty({x, y, vx, vy, theta, omega}) do
    xr = div(abs(x), 100_000)
    yr = div(abs(y), 100_000)
    vxr = div(abs(vx), 100_000)
    vyr = div(abs(vy), 100_000)
    tr = div(abs(theta), 100_000)
    wr = div(abs(omega), 100_000)
    xr + yr + vxr + vyr + tr + wr
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
      # Standard center drop
      {0, 141_000_000, 0, 0, 0, 0},
      # Matching Gymnasium's initial state distribution (vx±0.8, vy±0.5, ω±0.18)
      {0, 141_000_000, 60_000_000, 0, 0, 0},
      {0, 141_000_000, -60_000_000, 0, 0, 0},
      {0, 141_000_000, 0, -40_000_000, 0, 0},
      {0, 141_000_000, 0, 40_000_000, 0, 0},
      {0, 141_000_000, 0, 0, 0, 15_000_000},
      {0, 141_000_000, 0, 0, 0, -15_000_000},
      # Combined perturbations (common Gymnasium scenarios)
      {0, 141_000_000, 50_000_000, -30_000_000, 5_000_000, 10_000_000},
      {0, 141_000_000, -50_000_000, -30_000_000, -5_000_000, -10_000_000},
      {0, 141_000_000, 70_000_000, -20_000_000, -3_000_000, 12_000_000},
      {0, 141_000_000, -70_000_000, -20_000_000, 3_000_000, -12_000_000},
      # Larger angular perturbations (stress test tilt correction)
      {0, 141_000_000, 0, 0, 8_000_000, 0},
      {0, 141_000_000, 0, 0, -8_000_000, 0},
      {0, 141_000_000, 30_000_000, 0, 5_000_000, 10_000_000},
      {0, 141_000_000, -30_000_000, 0, -5_000_000, -10_000_000},
      # Near-ground scenarios
      {0, 50_000_000, 0, -20_000_000, 0, 0},
      {10_000_000, 70_000_000, 10_000_000, -15_000_000, 3_000_000, 5_000_000},
    ]
  end

  @impl true
  def state_to_list({x, y, vx, vy, theta, omega}), do: [x, y, vx, vy, theta, omega]

  @impl true
  def oracle_horizon(), do: 200

  @impl true
  def score_horizon(), do: 300
end
