defmodule Synthex.Gym.Mujoco do
  @moduledoc """
  Binary-weighted synthesis for MuJoCo continuous-action environments.

  Generalized version of Binary: bits_per_dim, n_action_dims, and
  dim names are runtime parameters rather than compile-time constants.
  Uses the shared mujoco.py oracle adapter.

  ## API surface

  Two ways to use this module:

    * `solve/2` — one-shot driver. Runs the full CEGAR loop in a single
      Elixir process. Good for laptop / local Python runs.

    * The resumable building blocks — `init_context/2`, `collect_states/2`,
      `build_features/2`, `optimize_bit/5`, `validate/3`, `shuffle_bits/2`,
      `seeds_for/3`. These are the same operations `solve/2` is built on,
      exposed so an external orchestrator (e.g. Oban jobs on Synthex Hub)
      can drive synthesis with persistent checkpoints between steps and
      auto-resume on crash.

  The resumable API is pure (no I/O or process state outside of the
  scorer it's handed), so it composes cleanly with any supervisor —
  Oban, GenServer, or a hand-rolled `Enum.reduce`.
  """

  alias Synthex.Gym.Oracle, as: GymOracle
  alias Synthex.Core.CEGIS

  @env_configs %{
    inverted_pendulum: %{
      gym_name: "InvertedPendulum-v5",
      n_action_dims: 1,
      num_dims: 4,
      max_steps: 1000,
      action_range: {-3.0, 3.0},
      dim_names: %{0 => "x", 1 => "x_dot", 2 => "theta", 3 => "theta_dot"},
      action_dim_names: %{0 => "force"}
    },
    inverted_double_pendulum: %{
      gym_name: "InvertedDoublePendulum-v5",
      n_action_dims: 1,
      # Gymnasium 1.x InvertedDoublePendulum-v5 obs (9):
      #   [cart_x, sin θ1, sin θ2, cos θ1, cos θ2,
      #    cart_v, θ1_dot, θ2_dot, qfrc_constraint[0]]
      num_dims: 9,
      max_steps: 1000,
      action_range: {-1.0, 1.0},
      dim_names: %{
        0 => "cart_x",
        1 => "sin_t1", 2 => "sin_t2",
        3 => "cos_t1", 4 => "cos_t2",
        5 => "cart_v", 6 => "t1_dot", 7 => "t2_dot",
        8 => "qfrc_c"
      },
      action_dim_names: %{0 => "force"}
    },
    swimmer: %{
      gym_name: "Swimmer-v5",
      n_action_dims: 2,
      num_dims: 8,
      max_steps: 1000,
      action_range: {-1.0, 1.0},
      dim_names: %{
        0 => "angle0", 1 => "angle1",
        2 => "vel_x", 3 => "vel_y",
        4 => "ang_vel0", 5 => "ang_vel1",
        6 => "ang_vel2", 7 => "ang_vel3"
      },
      action_dim_names: %{0 => "torque0", 1 => "torque1"}
    },
    hopper: %{
      gym_name: "Hopper-v5",
      n_action_dims: 3,
      num_dims: 11,
      max_steps: 1000,
      action_range: {-1.0, 1.0},
      dim_names: %{
        0 => "z", 1 => "angle",
        2 => "thigh_angle", 3 => "leg_angle", 4 => "foot_angle",
        5 => "vel_x", 6 => "vel_z",
        7 => "ang_vel", 8 => "thigh_vel", 9 => "leg_vel", 10 => "foot_vel"
      },
      action_dim_names: %{0 => "thigh", 1 => "leg", 2 => "foot"}
    },
    half_cheetah: %{
      gym_name: "HalfCheetah-v5",
      n_action_dims: 6,
      num_dims: 17,
      max_steps: 1000,
      action_range: {-1.0, 1.0},
      dim_names: (for i <- 0..16, into: %{}, do: {i, "d#{i}"}),
      action_dim_names: %{
        0 => "bthigh", 1 => "bshin", 2 => "bfoot",
        3 => "fthigh", 4 => "fshin", 5 => "ffoot"
      }
    },
    walker2d: %{
      gym_name: "Walker2d-v5",
      n_action_dims: 6,
      num_dims: 17,
      max_steps: 1000,
      action_range: {-1.0, 1.0},
      dim_names: (for i <- 0..16, into: %{}, do: {i, "d#{i}"}),
      action_dim_names: %{
        0 => "thigh_r", 1 => "leg_r", 2 => "foot_r",
        3 => "thigh_l", 4 => "leg_l", 5 => "foot_l"
      }
    },
    ant: %{
      gym_name: "Ant-v5",
      n_action_dims: 8,
      # Gymnasium 1.x default: qpos[2:]=13 + qvel=14 + cfrc_ext=78 = 105.
      num_dims: 105,
      max_steps: 1000,
      action_range: {-1.0, 1.0},
      dim_names: (for i <- 0..104, into: %{}, do: {i, "d#{i}"}),
      action_dim_names: %{
        0 => "hip_1", 1 => "ankle_1",
        2 => "hip_2", 3 => "ankle_2",
        4 => "hip_3", 5 => "ankle_3",
        6 => "hip_4", 7 => "ankle_4"
      }
    },
    humanoid: %{
      gym_name: "Humanoid-v5",
      n_action_dims: 17,
      # Gymnasium 1.x default Humanoid-v5: 348 obs dims
      # (qpos[2:]=22, qvel=23, cinert=140, cvel=84, qfrc_actuator=23, cfrc_ext=84−28=56).
      # Verify with: gym.make("Humanoid-v5").observation_space.shape
      num_dims: 348,
      # Humanoid actions are clipped to [-0.4, 0.4] in the standard task.
      max_steps: 1000,
      action_range: {-0.4, 0.4},
      dim_names: (for i <- 0..347, into: %{}, do: {i, "d#{i}"}),
      action_dim_names: %{
        0  => "abdomen_z",  1  => "abdomen_y",  2  => "abdomen_x",
        3  => "right_hip_x", 4 => "right_hip_z", 5  => "right_hip_y",
        6  => "right_knee",  7 => "left_hip_x",  8  => "left_hip_z",
        9  => "left_hip_y",  10 => "left_knee",
        11 => "right_shoulder1", 12 => "right_shoulder2", 13 => "right_elbow",
        14 => "left_shoulder1",  15 => "left_shoulder2",  16 => "left_elbow"
      }
    }
  }

  @doc """
  Look up the static config for an env (gym name, obs/action dims,
  action range, dim names, default max_steps).
  """
  @spec env_config(atom()) :: map()
  def env_config(env_key), do: Map.fetch!(@env_configs, env_key)

  @doc """
  List all envs this module knows how to synthesize for.
  """
  @spec known_envs() :: [atom()]
  def known_envs, do: Map.keys(@env_configs)

  @doc """
  Build a synthesis context from `env_key` + opts.

  Returns a `ctx` map carrying every parameter the resumable building
  blocks need: bits_per_dim, n_bits, depth, max_coeff, feature_types,
  scorer, etc. Same opts as `solve/2`; see that doc for the list.

  The returned ctx is the unit you'd checkpoint to disk / Postgres to
  resume a synthesis run across process restarts.
  """
  @spec init_context(atom(), keyword()) :: map()
  def init_context(env_key, opts \\ []) do
    cfg = Map.fetch!(@env_configs, env_key)
    bits_per_dim = Keyword.get(opts, :bits_per_dim, 3)
    n_action_dims = cfg.n_action_dims
    n_bits = bits_per_dim * n_action_dims
    weights = for i <- 0..(bits_per_dim - 1), do: Integer.pow(2, i)
    max_sum = Enum.sum(weights)

    max_steps = Keyword.get(opts, :max_steps, cfg.max_steps)
    depth = Keyword.get(opts, :depth, 1)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    tridiag_max_coeff = Keyword.get(opts, :tridiag_max_coeff, 2)
    tridiag_dims = Keyword.get(opts, :tridiag_dims, nil)
    n_episodes = Keyword.get(opts, :n_episodes, 30)
    top_k = Keyword.get(opts, :top_k, 20)
    max_iters = Keyword.get(opts, :max_iters, 5)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 3)
    feature_types = Keyword.get(opts, :feature_types, GymOracle.all_feature_types())

    # Pluggable oracle invocation. Defaults to `Synthex.Scoring.LocalPython`,
    # which forks a Python interpreter via `System.cmd`. To distribute,
    # plug in `Synthex.Hub.Scorer` from the synthex-hub client lib.
    scorer = Keyword.get(opts, :scorer) || Synthex.Scoring.default(env_key)

    %{
      env_key: env_key,
      cfg: cfg,
      bits_per_dim: bits_per_dim,
      n_bits: n_bits,
      n_action_dims: n_action_dims,
      weights: weights,
      max_sum: max_sum,
      max_steps: max_steps,
      depth: depth,
      max_coeff: max_coeff,
      tridiag_max_coeff: tridiag_max_coeff,
      tridiag_dims: tridiag_dims,
      n_episodes: n_episodes,
      top_k: top_k,
      max_iters: max_iters,
      cegar_rounds: cegar_rounds,
      feature_types: feature_types,
      scorer: scorer
    }
  end

  @doc """
  Initial predicate vector for a fresh run. All bits start as
  `:falsep` (every action dimension off → zero output).
  """
  @spec initial_predicates(map()) :: [Synthex.Core.PredProg.predicate()]
  def initial_predicates(ctx), do: List.duplicate(:falsep, ctx.n_bits)

  @doc """
  Run one CEGAR round's state collection. Calls the scorer's
  `collect_states` command with the current predicate vector and
  returns `{states, n_landings}`. Pure with respect to `ctx` and
  `preds`; the only side effect is whatever the scorer does
  (HTTP, Python subprocess, etc.).
  """
  @spec collect_states([Synthex.Core.PredProg.predicate()], map()) :: {[list()], non_neg_integer()}
  def collect_states(preds, ctx) do
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)

    request = %{
      "cmd" => "collect_states",
      "env_name" => ctx.cfg.gym_name,
      "bits_per_dim" => ctx.bits_per_dim,
      "bit_predicates" => serialized_preds,
      "seeds" => Enum.to_list(0..39),
      "max_steps" => ctx.max_steps
    }

    result = call_scorer!(request, ctx)
    {result["states"], result["n_landings"]}
  end

  @doc """
  Build the candidate feature set from observed states. Deterministic
  given `states` and `ctx`, so a worker that crashes mid-CEGAR-round
  can recompute features by re-running `collect_states` and this.
  """
  @spec build_features([list()], map()) :: [Synthex.Core.PredProg.predicate()]
  def build_features(states, ctx) do
    GymOracle.generate_features(states,
      env: ctx.env_key,
      n_dims: ctx.cfg.num_dims,
      max_coeff: ctx.max_coeff,
      tridiag_max_coeff: ctx.tridiag_max_coeff,
      tridiag_dims: ctx.tridiag_dims,
      feature_types: ctx.feature_types
    )
  end

  @doc """
  Produce a shuffled bit order for one CEGAR iteration. Pass a seed
  to make the shuffle deterministic across resumes; pass `nil` (the
  default) to let `:rand` pick its own state.

  Returned list always contains every bit index in `0..n_bits-1`
  exactly once.
  """
  @spec shuffle_bits(pos_integer(), term() | nil) :: [non_neg_integer()]
  def shuffle_bits(n_bits, seed \\ nil)

  def shuffle_bits(n_bits, nil) do
    Enum.to_list(0..(n_bits - 1)) |> Enum.shuffle()
  end

  def shuffle_bits(n_bits, seed) do
    # Assign each bit index a deterministic random sort key and sort
    # by it — a Fisher-Yates-equivalent shuffle whose output depends
    # only on `seed`. Reproducible across runs / nodes.
    {indexed, _} =
      Enum.map_reduce(0..(n_bits - 1), :rand.seed_s(:exsss, seed), fn i, rstate ->
        {k, rstate} = :rand.uniform_s(rstate)
        {{k, i}, rstate}
      end)

    indexed
    |> Enum.sort_by(fn {k, _i} -> k end)
    |> Enum.map(fn {_k, i} -> i end)
  end

  @doc """
  Deterministic episode-seed list for a given `(cegar_iter, iter)`
  pair. The CEGAR loop uses fresh seeds per iteration so a candidate
  predicate can't overfit to a single env reset; this function
  encodes the canonical offset scheme so a resumed iteration uses
  the SAME seeds it would have on the original attempt.
  """
  @spec seeds_for(pos_integer(), pos_integer(), map()) :: [non_neg_integer()]
  def seeds_for(cegar_iter, iter, ctx) do
    seed_offset = ((cegar_iter - 1) * ctx.max_iters + (iter - 1)) * ctx.n_episodes
    Enum.to_list(seed_offset..(seed_offset + ctx.n_episodes - 1))
  end

  @doc """
  Standard validation seed set. Held constant across all runs so
  validation scores are comparable.
  """
  @spec validation_seeds() :: [non_neg_integer()]
  def validation_seeds, do: Enum.to_list(10_000..10_199)

  @doc """
  Run one bit search: score every candidate (at depth 0 and, if
  `ctx.depth > 0`, the best depth-1 combinations) and return the
  improvement, if any.

  Returns:

    * `{:improved, new_pred, reward}` — `new_pred` beats `preds[bit_idx]`
      on `seeds`; reward is its mean episode return.
    * `:no_improvement` — no candidate strictly improves over the
      baseline.

  This is the unit of work an orchestrator should checkpoint between:
  the only state mutation is replacing `preds[bit_idx]` with `new_pred`
  on success, which the caller does itself.
  """
  @spec optimize_bit(
          [Synthex.Core.PredProg.predicate()],
          non_neg_integer(),
          [Synthex.Core.PredProg.predicate()],
          map(),
          [non_neg_integer()]
        ) ::
          {:improved, Synthex.Core.PredProg.predicate(), float()} | :no_improvement
  def optimize_bit(preds, bit_idx, features, ctx, seeds) do
    case do_optimize_bit(preds, bit_idx, features, ctx, seeds) do
      nil -> :no_improvement
      {new_pred, reward} -> {:improved, new_pred, reward}
    end
  end

  @doc """
  Run validation on a predicate vector. Returns `{total_reward,
  n_survived}` summed over `seeds`. Divide by `length(seeds)` for
  mean episode reward.
  """
  @spec validate([Synthex.Core.PredProg.predicate()], [non_neg_integer()], map()) ::
          {float(), non_neg_integer()}
  def validate(preds, seeds, ctx) do
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)

    request = %{
      "cmd" => "score_bit",
      "env_name" => ctx.cfg.gym_name,
      "bits_per_dim" => ctx.bits_per_dim,
      "candidates" => [],
      "bit_predicates" => serialized_preds,
      "target_bit" => 0,
      "seeds" => seeds,
      "max_steps" => ctx.max_steps
    }

    result = call_scorer!(request, ctx)
    {result["baseline_reward"], result["baseline_landings"]}
  end

  @doc """
  One-shot driver. Equivalent to building a context, picking an
  initial predicate vector, and running `cegar_rounds × max_iters`
  passes over the bits via `optimize_bit/5`. Emits a
  `[:synthex, :mujoco, :bit_accepted]` telemetry event on every
  accepted improvement.

  See module doc for opts.
  """
  def solve(env_key, opts \\ []) do
    ctx = init_context(env_key, opts)
    cfg = ctx.cfg

    {lo, hi} = cfg.action_range
    val_seeds = validation_seeds()

    IO.puts("  CSHRL Binary-Weighted Synthesis -- MuJoCo")
    IO.puts("  Env: #{cfg.gym_name}")
    IO.puts("  #{ctx.bits_per_dim} bits/dim x #{ctx.n_action_dims} dims = #{ctx.n_bits} predicates")
    IO.puts("  Obs dims: #{cfg.num_dims} | Action range: [#{lo}, #{hi}]")
    IO.puts("  Depth: #{ctx.depth}, Episodes: #{ctx.n_episodes}, TopK: #{ctx.top_k}")
    IO.puts("  Features: #{inspect(ctx.feature_types)} (max_coeff=#{ctx.max_coeff}, tridiag_max_coeff=#{ctx.tridiag_max_coeff})\n")

    bit_preds = initial_predicates(ctx)

    final_preds =
      Enum.reduce(1..ctx.cegar_rounds, bit_preds, fn cegar_iter, preds ->
        IO.puts("\n  CEGAR Round #{cegar_iter}/#{ctx.cegar_rounds}")

        {states, _n_succ} = collect_states(preds, ctx)
        IO.puts("  #{length(states)} states collected")

        features = build_features(states, ctx)
        IO.puts("  #{length(features)} features\n")

        Enum.reduce(1..ctx.max_iters, preds, fn iter, cur_preds ->
          IO.puts("\n  Iteration #{iter}/#{ctx.max_iters}")
          seeds = seeds_for(cegar_iter, iter, ctx)
          bit_indices = shuffle_bits(ctx.n_bits)

          new_preds =
            Enum.reduce(bit_indices, cur_preds, fn bit_idx, ps ->
              dim_idx = div(bit_idx, ctx.bits_per_dim)
              bit_pos = rem(bit_idx, ctx.bits_per_dim)
              weight = Enum.at(ctx.weights, bit_pos)
              dim_name = cfg.action_dim_names[dim_idx] || "dim#{dim_idx}"

              IO.puts("\n  >> Bit #{bit_idx}: #{dim_name} weight=#{weight}")

              case optimize_bit(ps, bit_idx, features, ctx, seeds) do
                :no_improvement ->
                  IO.puts("    No improvement")
                  ps

                {:improved, new_pred, reward} ->
                  IO.puts("    reward=#{Float.round(reward, 1)}")
                  updated = List.replace_at(ps, bit_idx, new_pred)
                  emit_bit_accepted(ctx, cegar_iter, iter, bit_idx, reward, updated)
                  updated
              end
            end)

          {val_reward, val_survived} = validate(new_preds, val_seeds, ctx)
          avg = Float.round(val_reward / length(val_seeds), 1)
          IO.puts("  Validation: avg=#{avg}/ep survived=#{val_survived}/#{length(val_seeds)}")
          new_preds
        end)
      end)

    IO.puts("\n  SYNTHESIS COMPLETE -- #{cfg.gym_name}")
    {val_reward, _} = validate(final_preds, val_seeds, ctx)
    avg = Float.round(val_reward / length(val_seeds), 1)
    IO.puts("  Final validation avg: #{avg}/ep")

    final_preds
  end

  @doc """
  Emit the `[:synthex, :mujoco, :bit_accepted]` telemetry event for
  an accepted improvement. Public so external orchestrators (Synthex
  Hub's Oban master) can fire the same event from outside `solve/2`,
  keeping telemetry handlers (snapshot publishers, dashboards, ...)
  working uniformly across in-process and distributed runs.
  """
  @spec emit_bit_accepted(
          map(),
          pos_integer(),
          pos_integer(),
          non_neg_integer(),
          float(),
          [Synthex.Core.PredProg.predicate()]
        ) :: :ok
  def emit_bit_accepted(ctx, cegar_iter, iter, bit_idx, reward, updated_preds) do
    :telemetry.execute(
      [:synthex, :mujoco, :bit_accepted],
      %{reward: reward},
      %{
        env_key: ctx.env_key,
        env_name: ctx.cfg.gym_name,
        cegar_iter: cegar_iter,
        iter: iter,
        bit_idx: bit_idx,
        bits_per_dim: ctx.bits_per_dim,
        n_action_dims: ctx.n_action_dims,
        n_bits: ctx.n_bits,
        action_range: ctx.cfg.action_range,
        action_dim_names: ctx.cfg.action_dim_names,
        bit_predicates: updated_preds
      }
    )
  end

  # ── Internals ─────────────────────────────────────────────────

  defp do_optimize_bit(preds, bit_idx, features, ctx, seeds) do
    atoms = CEGIS.enumerate(features, 0)
    all_d0 = [:truep, :falsep | atoms]

    {scored_d0, baseline} = score_bit_candidates(all_d0, preds, bit_idx, seeds, ctx)
    best_d0 = Enum.max_by(scored_d0, fn {_i, r, _l} -> r end, fn -> nil end)

    d0_result =
      case best_d0 do
        nil -> nil
        {idx, reward, _count} ->
          if reward > baseline do
            {Enum.at(all_d0, idx), reward, scored_d0}
          else
            nil
          end
      end

    if ctx.depth == 0 do
      case d0_result do
        nil -> nil
        {p, r, _} -> {p, r}
      end
    else
      top_atoms =
        scored_d0
        |> Enum.sort_by(fn {_idx, r, _l} -> -r end)
        |> Enum.take(ctx.top_k)
        |> Enum.map(fn {idx, _r, _l} -> Enum.at(all_d0, idx) end)
        |> Enum.reject(fn p -> p == :truep or p == :falsep end)

      negations = Enum.map(top_atoms, fn p -> {:not, p} end)
      d1_candidates =
        (for p <- top_atoms, q <- top_atoms, p != q, do: {:and, p, q}) ++
        (for p <- top_atoms, q <- top_atoms, p != q, do: {:or, p, q}) ++
        (for p <- negations, q <- top_atoms, do: {:and, p, q})
      d1_candidates = Enum.uniq(d1_candidates)

      {scored_d1, _} = score_bit_candidates(d1_candidates, preds, bit_idx, seeds, ctx)
      best_d1 = Enum.max_by(scored_d1, fn {_i, r, _l} -> r end, fn -> nil end)

      threshold = case d0_result do
        nil -> baseline
        {_, r, _} -> r
      end

      case best_d1 do
        nil ->
          case d0_result do
            nil -> nil
            {p, r, _} -> {p, r}
          end
        {idx, reward, _count} ->
          if reward > threshold do
            {Enum.at(d1_candidates, idx), reward}
          else
            case d0_result do
              nil -> nil
              {p, r, _} -> {p, r}
            end
          end
      end
    end
  end

  defp score_bit_candidates(candidates, preds, target_bit, seeds, ctx) do
    serialized_candidates = Enum.map(candidates, &GymOracle.serialize_pred/1)
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)

    request = %{
      "cmd" => "score_bit",
      "env_name" => ctx.cfg.gym_name,
      "bits_per_dim" => ctx.bits_per_dim,
      "candidates" => serialized_candidates,
      "bit_predicates" => serialized_preds,
      "target_bit" => target_bit,
      "seeds" => seeds,
      "max_steps" => ctx.max_steps
    }

    result = call_scorer!(request, ctx)

    scored =
      Enum.map(result["scores"] || [], fn s ->
        {s["idx"], s["reward"], Map.get(s, "landings", 0)}
      end)

    {scored, result["baseline_reward"]}
  end

  # Single point of contact with the outside world. The scorer is
  # whatever the caller plugged in via `init_context(..., scorer: ...)`;
  # `Synthex.Scoring.LocalPython` is the default. Distributed scorers
  # (e.g. `Synthex.Hub.Scorer` from synthex-hub-client) live outside
  # this repo so synthex itself stays HTTP-free.
  defp call_scorer!(request, ctx) do
    case ctx.scorer.(request) do
      {:ok, response} ->
        response

      {:error, reason} ->
        raise "Synthex.Gym.Mujoco scorer failed (cmd=#{request["cmd"]}): #{reason}"
    end
  end
end
