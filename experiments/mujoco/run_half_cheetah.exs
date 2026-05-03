# HalfCheetah experiment
#
# Usage: cd /Users/rcc/Research/synthex && mix run experiments/mujoco/run_half_cheetah.exs

Synthex.Gym.Mujoco.solve(:half_cheetah,
  bits_per_dim: 3,
  depth: 1,
  max_coeff: 5,
  feature_types: [:axis, :diag],
  n_episodes: 50,
  top_k: 24,
  max_iters: 8,
  cegar_rounds: 8,
  max_steps: 1000
)
