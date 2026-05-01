# Experiments

Each subdirectory contains reproducible experiment configurations for one environment.

## Running experiments

**Via TOML config (recommended):**
```bash
mix run -e 'Synthex.Experiment.run("experiments/tetris/successor_hybrid.toml")'
```

**Via .exs script:**
```bash
mix run experiments/lunarlander/chain_d1.exs
```

**Long-running (background):**
```bash
nohup mix run -e 'Synthex.Experiment.run("experiments/tetris/successor_long.toml")' \
  > results/tetris_long.log 2>&1 &
echo $! > tetris_long.pid
```

## TOML config format

```toml
[experiment]
name = "experiment_name"    # Used for results directory
method = "chain"            # chain | ranking | successor | swapnet | binary | mujoco | pairwise_matrix | pure_chain

[environment]
env = "lunarlander"         # Environment key (matches Synthex.Gym.Oracle registry)
actions = ["fire_left", "fire_right", "fire_main"]  # Action priority (chain methods)
default = "do_nothing"      # Default/fallback action

[synthesis]
depth = 1                   # Boolean predicate depth (0 = atoms only, 1 = AND/OR)
max_coeff = 5               # Max diagonal coefficient
n_episodes = 200            # Episodes per candidate evaluation
top_k = 30                  # Depth-0 atoms kept for depth-1 exploration
max_iters = 5               # Coordinate descent iterations per CEGAR round
cegar_rounds = 3            # CEGAR abstraction refinement rounds
max_steps = 1000            # Max steps per episode

[successor]                 # Only for method = "successor"
lookahead = 200             # Rollout steps for successor quality estimation
sample_every = 5            # Sample every N steps for successor data
succ_top_k = 200            # Candidates surviving successor pre-filter

[validation]
val_episodes = 500          # Episodes for validation scoring
```

## Results

Results are saved to `results/<experiment_name>/<timestamp>/` containing:
- `config.toml` — copy of the experiment config
- `policy.json` — synthesized policy chain
- `git_sha.txt` — git commit hash for reproducibility
