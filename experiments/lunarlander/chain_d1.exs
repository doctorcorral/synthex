# LunarLander-v3: CEGAR + Coordinate Descent chain synthesis
#
# Priority: FireLeft > FireRight > FireMain > DoNothing
# 6D state: (x, y, vx, vy, theta, omega)
#
# Usage: mix run experiments/lunarlander/chain_d1.exs

{chain, default} = Synthex.Gym.Chain.solve(
  [:fire_left, :fire_right, :fire_main],
  :do_nothing,
  depth: 1,
  max_coeff: 5,
  n_episodes: 200,
  top_k: 30,
  max_iters: 5,
  cegar_rounds: 3,
  max_steps: 1000
)

Synthex.Gym.Oracle.print_deployable(chain, default, :lunarlander)
