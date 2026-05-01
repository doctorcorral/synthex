defmodule Synthex.Envs.Pendulum do
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

  def fclip(lo, hi, x) do
    cond do
      x < lo -> lo
      x > hi -> hi
      true -> x
    end
  end

  def act_f(:torque_neg), do: -200_000_000
  def act_f(:no_torque), do: 0
  def act_f(:torque_pos), do: 200_000_000

  @dt 5_000_000
  @omega_max 800_000_000

  @impl true
  def step({theta, omega}, a) do
    grav = f_mul(1_500_000_000, sin7(theta))
    u = f_mul(300_000_000, act_f(a))
    acc = grav + u
    
    omega_prime = fclip(-@omega_max, @omega_max, omega + f_mul(acc, @dt))
    theta_prime = theta + f_mul(omega_prime, @dt)
    
    # Wrap theta between -pi and pi (-314159265, 314159265)
    pi = 314_159_265
    two_pi = 2 * pi
    
    wrapped_theta = rem(theta_prime + pi, two_pi)
    wrapped_theta = if wrapped_theta < 0, do: wrapped_theta + two_pi, else: wrapped_theta
    final_theta = wrapped_theta - pi
    
    {final_theta, omega_prime}
  end

  @impl true
  def terminal?(_), do: false

  @impl true
  def penalty({theta, omega}) do
    th_sd = div_int(abs(theta), 1_000_000)
    om_sd = div_int(abs(omega), 1_000_000)
    
    th_sd * th_sd + div_int(om_sd * om_sd, 10)
  end

  @impl true
  def crash_penalty(), do: 0 # No early termination crash

  @impl true
  def max_penalty(), do: 10_000_000

  @impl true
  def actions(), do: [:torque_neg, :no_torque, :torque_pos]

  @impl true
  def starts do
    [
      # Down (hardest)
      {314_159_265, 0},
      {-314_159_265, 0},
      # Horizontal
      {157_079_632, 0},
      {-157_079_632, 0},
      # Slightly off top
      {10_000_000, 0},
      {-10_000_000, 0}
    ]
  end

  @impl true
  def state_to_list({theta, omega}), do: [theta, omega]

  @impl true
  def oracle_horizon(), do: 100

  @impl true
  def score_horizon(), do: 200
end
