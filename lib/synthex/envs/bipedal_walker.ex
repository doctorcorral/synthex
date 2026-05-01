defmodule Synthex.Envs.BipedalWalker do
  @behaviour Synthex.Environment

  @doc """
  BipedalWalker has a 24-dimensional continuous state space and a 4-dimensional
  continuous action space [-1, 1].
  We discretize it into 6 macro actions to make it amenable to CSHRL synthesis.
  """
  
  # State variables (24 dims): hull angle, angularVelocity, x velocity, y velocity, 
  # joints angles & speeds, legs contact with ground, 10 lidar rangefinder readings.
  
  @impl true
  def actions() do
    [
      :do_nothing,
      :left_leg_forward,
      :left_leg_backward,
      :right_leg_forward,
      :right_leg_backward,
      :both_legs_forward
    ]
  end

  @impl true
  def starts() do
    # BipedalWalker starts are mostly flat with slight variations in angle/velocity.
    # Scaled by 100M for fixed-point math.
    [
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {100_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-100_000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    ]
  end

  @impl true
  def state_to_list(state) do
    Tuple.to_list(state)
  end

  @impl true
  def terminal?(_state), do: false # Requires a physics simulator

  @impl true
  def step(_state, _action) do
    # Like other environments, complex physics (Box2D) can't be easily replicated in Elixir.
    # In a full setup, this would call out to a Python/C++ physics engine, or we rely on 
    # Python-side data collection for the dataset and do purely offline Imitation.
    raise "BipedalWalker step function requires external physics simulator."
  end

  @impl true
  def penalty(_state) do
    # Proxy penalty
    0
  end

  @impl true
  def max_penalty(), do: 1_000_000_000

  @impl true
  def crash_penalty(), do: 500_000_000

  @impl true
  def oracle_horizon(), do: 100

  @impl true
  def score_horizon(), do: 200
end
