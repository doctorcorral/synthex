# Synthex

A high-performance concurrent synthesis engine for **Coinductive Symmetric Homomorphism Reinforcement Learning (CSHRL)**.

Synthex discovers formally verifiable control policies expressed as Boolean predicate programs by aggressively searching the candidate space across CPU cores. The synthesized predicates can be verified in Agda, guaranteeing mathematical correctness.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Experiment Runner               │
│          (TOML config → method dispatch)         │
├────────────────────┬────────────────────────────┤
│    Pure Elixir     │     Gymnasium-in-the-loop   │
│   (no Python)      │        (Python IPC)         │
│                    │                             │
│  Oracle            │  Gym.Chain                  │
│  StateCEGAR        │  Gym.Ranking               │
│  PairwiseMatrix    │  Gym.Successor             │
│  Chain             │  Gym.SwapNetwork            │
│  Permutation.*     │  Gym.Binary                │
│                    │  Gym.Mujoco                │
│                    │  Gym.Permutation            │
├────────────────────┴────────────────────────────┤
│                  Core Foundation                 │
│  PredProg · CEGIS · ContinuousFeatures · Env    │
├─────────────────────────────────────────────────┤
│              Python Oracle Adapters              │
│  lunarlander · pendulum · tetris · pong · ...    │
└─────────────────────────────────────────────────┘
```

### Core (`lib/synthex/core/`)

- **`PredProg`** — Boolean predicate AST: `truep`, `falsep`, `feat`, `not`, `and`, `or`. Mirrors the Agda data type.
- **`CEGIS`** — Depth-bounded enumeration of the predicate version space + concurrent refinement.
- **`ContinuousFeatures`** — Axis-aligned and diagonal boundary feature generation from trajectories.
- **`Environment`** — Behaviour for pure Elixir environment implementations.

### Pure Synthesis (`lib/synthex/pure/`)

Synthesis that runs entirely in Elixir against `Environment` implementations (no Python needed):

- **`Oracle`** — Multi-step rollout oracle for pairwise action comparison.
- **`StateCEGAR`** — CEGIS inner loop + CEGAR outer loop with Flow parallelism.
- **`PairwiseMatrix`** — Full N(N-1)/2 pairwise relation matrix synthesis.
- **`Chain`** — Priority-chain synthesis (cycle-free by construction).
- **`Permutation.*`** — Rank-tree and permutation policy synthesis subsystem.

### Gym Synthesis (`lib/synthex/gym/`)

Synthesis using real Gymnasium environments via Python IPC:

- **`Oracle`** — Python bridge with declarative environment registry, feature generation, CEGAR refinement.
- **`Chain`** — CEGAR + coordinate descent with Gymnasium episode scoring.
- **`Ranking`** — Flat partition CEGAR for ranking lists.
- **`Successor`** — Hybrid synthesis using CoinductiveHomomorphism (successor-state quality pre-filter + episode validation).
- **`SwapNetwork`** — Sorting-network conditional swap synthesis.
- **`Binary`** / **`Mujoco`** — Bit-decomposition for continuous action spaces.

### Python Oracles (`oracles/`)

Thin adapters that interact with Gymnasium environments. Each oracle:
- Receives JSON requests from Elixir (collect states, score candidates, etc.)
- Runs episodes in the actual environment
- Returns results as JSON

All oracles share `base_oracle.py` for predicate evaluation and the IPC protocol.

## Quick Start

### Prerequisites

- **Elixir** >= 1.18
- **Python** >= 3.10 with: `pip install -r oracles/requirements.txt`
- **ALE** (for Atari environments): included via `ale-py`

### Installation

```bash
git clone <repo-url> synthex
cd synthex
mix deps.get
pip install -r oracles/requirements.txt
```

### Running an experiment

**Config-driven:**
```bash
mix run -e 'Synthex.Experiment.run("experiments/lunarlander/chain_d1.toml")'
```

**Script-driven:**
```bash
mix run experiments/lunarlander/chain_d1.exs
```

**Pure Elixir (no Python):**
```bash
mix run -e 'Synthex.Pure.PairwiseMatrix.solve(Synthex.Envs.MountainCar, 1, 3, 100)'
```

**Long-running (background):**
```bash
nohup mix run -e 'Synthex.Experiment.run("experiments/tetris/successor_long.toml")' \
  > results/tetris_long.log 2>&1 &
echo $! > tetris_long.pid
```

### Running tests

```bash
mix test
```

## Adding a New Environment

### 1. Python oracle adapter

Create `oracles/my_env.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from base_oracle import eval_feature, eval_pred, chain_action, run_oracle, score_batch_parallel
import gymnasium as gym
import numpy as np

def make_env(seed=None):
    env = gym.make("MyEnv-v1")
    if seed is not None:
        env.reset(seed=seed)
    return env

def extract_state(obs):
    return [float(obs[0]), float(obs[1]), ...]  # N-dimensional state vector

def dispatch(request):
    cmd = request["cmd"]
    if cmd == "collect_states":
        # Run episodes, collect states
        ...
    elif cmd == "score":
        # Score candidate predicates
        ...
    return result

if __name__ == "__main__":
    run_oracle(dispatch)
```

### 2. Register in Elixir

Add an entry to the `@envs` map in `lib/synthex/gym/oracle.ex`:

```elixir
my_env: %{
  actions: %{action_a: 0, action_b: 1, action_c: 2},
  oracle: "my_env.py",
  dims: 3,
  dim_names: %{0 => "x", 1 => "y", 2 => "z"},
  dim_py: %{0 => "x", 1 => "y", 2 => "z"},
  obs_unpack: "    x, y, z = obs[:3]"
}
```

### 3. Create experiment config

```toml
[experiment]
name = "my_env_chain"
method = "chain"

[environment]
env = "my_env"
actions = ["action_a", "action_b"]
default = "action_c"

[synthesis]
depth = 1
max_coeff = 5
n_episodes = 30
top_k = 20
max_iters = 5
cegar_rounds = 3
max_steps = 500
```

## Supported Environments

| Environment | Type | Actions | State Dims | Oracle |
|-------------|------|---------|-----------|--------|
| LunarLander-v3 | Classic Control | 4 | 6 | `lunarlander.py` |
| Pendulum-v1 | Classic Control | 3 | 3 | `pendulum.py` |
| CartPole-v1 | Classic Control | 2 | 4 | `cartpole.py` |
| Acrobot-v1 | Classic Control | 3 | 6 | `acrobot.py` |
| MountainCar-v0 | Classic Control | 3 | 2 | `mountaincar.py` |
| BipedalWalker-v3 | Box2D | 2 (binary) | 24 | `bipedal.py` |
| ALE/Pong-v5 | Atari (RAM) | 3 | 6 | `pong.py` |
| ALE/Breakout-v5 | Atari (RAM) | 3 | 5 | `breakout.py` |
| ALE/Tetris-v5 | Atari (RAM) | 5 | 8/23 | `tetris.py` |
| InvertedPendulum-v5 | MuJoCo | 2 (binary) | 4 | `mujoco.py` |
| Swimmer-v5 | MuJoCo | 2 (binary) | 8 | `mujoco.py` |
| CliffWalking-v1 | Tabular | 4 | 2 | `cliffwalking.py` |

## Integrating with Agda

Synthex outputs predicate programs that can be translated into Agda's `PredProg` DSL:

```
fire_left  when  (x<0.1 ∧ vy<-0.5)
fire_main  when  (y<0.8 ∧ 3·vx+vy<0)
do_nothing otherwise
```

translates to:

```agda
chain : ChainPolicy 4
chain = record
  { p₁ = and (axis 0 0.1) (axis 3 -0.5) ; a₁ = fire-left
  ; p₂ = and (axis 1 0.8) (diag 2 3 3)  ; a₂ = fire-main
  ; default = do-nothing
  }
```

The Agda typechecker verifies the `preserves` field of the `CoindHomo` relation,
guaranteeing the policy satisfies the coinductive homomorphism property.

## License

MIT
