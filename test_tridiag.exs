defmodule TestTridiag do
  def run do
    # Using 4 dims so tridiag has enough dimensions (requires at least 3)
    states = [[0.0, 0.0, 0.0, 0.0]]
    max_coeff = 2
    
    # We have to mock num_dims for a dummy env or just use an existing one
    opts = [
      env: :inverted_pendulum, # has 4 dims
      max_coeff: max_coeff,
      feature_types: [:tridiag]
    ]
    
    feats = Synthex.GymOracle.generate_features(states, opts)
    IO.puts("Generated #{length(feats)} tridiagonal features for 4 dims with max_coeff=#{max_coeff}")
  end
end

TestTridiag.run()
