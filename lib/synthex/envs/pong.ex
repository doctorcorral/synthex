defmodule Synthex.Envs.Pong do
  @behaviour Synthex.Environment
  
  @scale 100_000_000
  @paddle_speed 5_000_000
  
  def clamp_y(y) do
    cond do
      y < 0 -> 0
      y > @scale -> @scale
      true -> y
    end
  end

  def move_paddle(py, :up), do: clamp_y(py - @paddle_speed)
  def move_paddle(py, :stay), do: py
  def move_paddle(py, :down), do: clamp_y(py + @paddle_speed)

  @impl true
  def step({bx, by, vx, vy, py}, a) do
    py_prime = move_paddle(py, a)
    bx_prime = bx + vx
    by_temp = by + vy
    
    bounce_top = by_temp < 0
    bounce_bot = by_temp > @scale
    
    {by_prime, vy_prime} =
      cond do
        bounce_top -> {-by_temp, -vy}
        bounce_bot -> {@scale - (by_temp - @scale), -vy}
        true -> {by_temp, vy}
      end
      
    {bx_prime, by_prime, vx, vy_prime, py_prime}
  end

  @impl true
  def terminal?({bx, _, _, _, _}) do
    bx < 0 or bx > @scale
  end

  @impl true
  def penalty({_, by, _, _, py}), do: abs(by - py)

  @impl true
  def crash_penalty(), do: 99_999_999

  @impl true
  def max_penalty(), do: 100_000_000

  @impl true
  def actions(), do: [:up, :stay, :down]

  @impl true
  def starts do
    [
      {50_000_000, 50_000_000, 2_000_000, 1_000_000, 50_000_000},
      {50_000_000, 20_000_000, 2_000_000, -1_000_000, 50_000_000},
      {50_000_000, 80_000_000, 2_000_000, 1_000_000, 50_000_000}
    ]
  end

  @impl true
  def state_to_list({bx, by, vx, vy, py}), do: [bx, by, vx, vy, py]

  @impl true
  def oracle_horizon(), do: 2 # Fast paddle = instantaneous reaction

  @impl true
  def score_horizon(), do: 50
end
