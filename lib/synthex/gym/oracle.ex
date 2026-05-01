defmodule Synthex.Gym.Oracle do
  @moduledoc """
  Bridge to Python Gymnasium adapters + all learning logic for Gym-in-the-loop synthesis.

  Python scripts are thin adapters that only interact with the environment.
  All feature generation, CEGAR counterexample detection, and abstraction
  refinement logic lives here in Synthex.
  """

  # ── Declarative Environment Registry ────────────────────────────

  @envs %{
    lunarlander: %{
      actions: %{do_nothing: 0, fire_left: 1, fire_main: 2, fire_right: 3},
      oracle: "lunarlander.py",
      dims: 6,
      dim_names: %{0 => "x", 1 => "y", 2 => "vx", 3 => "vy", 4 => "θ", 5 => "ω"},
      dim_py: %{0 => "x", 1 => "y", 2 => "vx", 3 => "vy", 4 => "theta", 5 => "omega"},
      obs_unpack: "    x, y, vx, vy, theta, omega = obs[:6]"
    },
    pendulum: %{
      actions: %{torque_neg: 0, no_torque: 1, torque_pos: 2},
      oracle: "pendulum.py",
      dims: 3,
      dim_names: %{0 => "cosθ", 1 => "sinθ", 2 => "ω"},
      dim_py: %{0 => "cos_theta", 1 => "sin_theta", 2 => "omega"},
      obs_unpack: "    cos_theta, sin_theta, omega = obs[:3]"
    },
    pong: %{
      actions: %{up: 0, noop: 1, down: 2},
      oracle: "pong.py",
      dims: 6,
      dim_names: %{0 => "bx", 1 => "by", 2 => "py", 3 => "vx", 4 => "vy", 5 => "ey"},
      dim_py: %{0 => "ball_x", 1 => "ball_y", 2 => "player_y", 3 => "ball_vx", 4 => "ball_vy", 5 => "enemy_y"},
      obs_unpack: "    ball_x, ball_y, player_y, ball_vx, ball_vy, enemy_y = obs[:6]"
    },
    cartpole: %{
      actions: %{left: 0, right: 1},
      oracle: "cartpole.py",
      dims: 4,
      dim_names: %{0 => "x", 1 => "ẋ", 2 => "θ", 3 => "θ̇"},
      dim_py: %{0 => "x", 1 => "x_dot", 2 => "theta", 3 => "theta_dot"},
      obs_unpack: "    x, x_dot, theta, theta_dot = obs[:4]"
    },
    acrobot: %{
      actions: %{torque_neg: 0, no_torque: 1, torque_pos: 2},
      oracle: "acrobot.py",
      dims: 6,
      dim_names: %{0 => "c1", 1 => "s1", 2 => "c2", 3 => "s2", 4 => "ω1", 5 => "ω2"},
      dim_py: %{0 => "cos1", 1 => "sin1", 2 => "cos2", 3 => "sin2", 4 => "w1", 5 => "w2"},
      obs_unpack: "    cos1, sin1, cos2, sin2, w1, w2 = obs[:6]"
    },
    mountaincar: %{
      actions: %{push_left: 0, no_push: 1, push_right: 2},
      oracle: "mountaincar.py",
      dims: 2,
      dim_names: %{0 => "pos", 1 => "vel"},
      dim_py: %{0 => "pos", 1 => "vel"},
      obs_unpack: "    pos, vel = obs[:2]"
    },
    breakout: %{
      actions: %{right: 0, noop: 1, left: 2},
      oracle: "breakout.py",
      dims: 5,
      dim_names: %{0 => "bx", 1 => "by", 2 => "px", 3 => "dx", 4 => "dy"},
      dim_py: %{0 => "ball_x", 1 => "ball_y", 2 => "paddle_x", 3 => "ball_dx", 4 => "ball_dy"},
      obs_unpack: "    ball_x, ball_y, paddle_x, ball_dx, ball_dy = obs[:5]"
    },
    cliffwalking: %{
      actions: %{up: 0, right: 1, down: 2, left: 3},
      oracle: "cliffwalking.py",
      dims: 2,
      dim_names: %{0 => "row", 1 => "col"},
      dim_py: %{0 => "row", 1 => "col"},
      obs_unpack: "    row, col = obs // 12, obs % 12"
    },
    bipedal: %{
      actions: %{bit_off: 0, bit_on: 1},
      oracle: "bipedal.py",
      dims: 24,
      dim_names: %{},
      dim_py: %{},
      obs_unpack: "    # 24-dimensional BipedalWalker observation"
    },
    inverted_pendulum: %{
      actions: %{bit_off: 0, bit_on: 1},
      oracle: "mujoco.py",
      dims: 4,
      dim_names: %{},
      dim_py: %{},
      obs_unpack: "    # InvertedPendulum observation",
      mujoco_env: "InvertedPendulum-v5"
    },
    swimmer: %{
      actions: %{bit_off: 0, bit_on: 1},
      oracle: "mujoco.py",
      dims: 8,
      dim_names: %{},
      dim_py: %{},
      obs_unpack: "    # Swimmer observation",
      mujoco_env: "Swimmer-v5"
    },
    hopper: %{
      actions: %{bit_off: 0, bit_on: 1},
      oracle: "mujoco.py",
      dims: 11,
      dim_names: %{},
      dim_py: %{},
      obs_unpack: "    # Hopper observation",
      mujoco_env: "Hopper-v5"
    },
    half_cheetah: %{
      actions: %{bit_off: 0, bit_on: 1},
      oracle: "mujoco.py",
      dims: 17,
      dim_names: %{},
      dim_py: %{},
      obs_unpack: "    # HalfCheetah observation",
      mujoco_env: "HalfCheetah-v5"
    },
    walker2d: %{
      actions: %{bit_off: 0, bit_on: 1},
      oracle: "mujoco.py",
      dims: 17,
      dim_names: %{},
      dim_py: %{},
      obs_unpack: "    # Walker2d observation",
      mujoco_env: "Walker2d-v5"
    },
    tetris: %{
      actions: %{left: 0, right: 1, rotate: 2, down: 3, noop: 4},
      oracle: "tetris.py",
      dims: 8,
      dim_names: %{0 => "px", 1 => "py", 2 => "rot", 3 => "vis",
                   4 => "fill", 5 => "bot", 6 => "rows", 7 => "pcs"},
      dim_py: %{0 => "piece_x", 1 => "piece_y", 2 => "rot_raw", 3 => "piece_vis",
                4 => "board_fill", 5 => "bottom_fill", 6 => "occupied_rows", 7 => "pieces_placed"},
      obs_unpack: "    piece_x, piece_y, rot_raw, piece_vis, board_fill, bottom_fill, occupied_rows, pieces_placed = obs[:8]"
    },
    tetris_v2: %{
      actions: %{left: 0, right: 1, rotate: 2, down: 3, noop: 4},
      oracle: "tetris_v2.py",
      dims: 23,
      dim_names: %{0 => "r0", 1 => "r1", 2 => "r2", 3 => "r3", 4 => "r4",
                   5 => "r5", 6 => "r6", 7 => "r7", 8 => "r8", 9 => "r9",
                   10 => "r10", 11 => "r11", 12 => "r12", 13 => "r13",
                   14 => "r14", 15 => "r15", 16 => "r16", 17 => "r17",
                   18 => "r18", 19 => "r19", 20 => "px", 21 => "py", 22 => "rot"},
      dim_py: %{},
      obs_unpack: "    # 23-dimensional Tetris v2 observation"
    }
  }

  def envs, do: @envs

  def action_map(env), do: @envs[env].actions
  def oracle_script(env), do: Path.join(oracles_dir(), @envs[env].oracle)
  def num_dims(env), do: @envs[env].dims
  def mujoco_env_name(env), do: Map.get(@envs[env] || %{}, :mujoco_env)

  defp oracles_dir do
    Application.get_env(:synthex, :oracles_dir, Path.expand("../../../oracles", __DIR__))
  end

  defp project_root do
    Application.get_env(:synthex, :project_root, Path.expand("../../..", __DIR__))
  end

  # ── Public API: Environment Interaction ────────────────────────

  @doc "Run episodes in Gymnasium, return raw states visited."
  def get_trajectory_states(chain, default_action, opts \\ []) do
    env = Keyword.get(opts, :env, :lunarlander)
    seeds = Keyword.get(opts, :seeds, Enum.to_list(0..39))
    max_steps = Keyword.get(opts, :max_steps, 300)

    request = %{
      "cmd" => "collect_states",
      "chain" => serialize_chain(chain, env),
      "default" => serialize_action(default_action, env),
      "seeds" => seeds,
      "max_steps" => max_steps
    }

    result = call_python(request, env)
    {result["states"], result["n_stabilized"] || result["n_landings"] || 0}
  end

  @doc """
  Run episodes with multi-step lookahead.
  At sampled steps, tries each action then follows the policy for
  `lookahead` steps total. Returns per-step [state, chosen, %{aid => N-step reward}].
  """
  def explore(chain, default_action, opts \\ []) do
    env = Keyword.get(opts, :env, :pendulum)
    seeds = Keyword.get(opts, :seeds, Enum.to_list(0..49))
    max_steps = Keyword.get(opts, :max_steps, 200)
    lookahead = Keyword.get(opts, :lookahead, 20)

    request = %{
      "cmd" => "explore",
      "chain" => serialize_chain(chain, env),
      "default" => serialize_action(default_action, env),
      "seeds" => seeds,
      "max_steps" => max_steps,
      "lookahead" => lookahead
    }

    result = call_python(request, env)
    {result["steps"], result["n_stabilized"], result["n_episodes"]}
  end

  @doc "Score candidate predicates against Gymnasium."
  def score_candidates(candidates, stage_action, default_action, chain_so_far, opts \\ []) do
    env = Keyword.get(opts, :env, :lunarlander)
    seeds = Keyword.get(opts, :seeds, Enum.to_list(0..29))
    chain_after = Keyword.get(opts, :chain_after, [])
    max_steps = Keyword.get(opts, :max_steps, 300)

    serialized_candidates = Enum.map(candidates, &serialize_pred/1)

    request = %{
      "cmd" => "score",
      "candidates" => serialized_candidates,
      "stage_action" => serialize_action(stage_action, env),
      "default" => serialize_action(default_action, env),
      "chain_so_far" => serialize_chain(chain_so_far, env),
      "chain_after" => serialize_chain(chain_after, env),
      "seeds" => seeds,
      "max_steps" => max_steps
    }

    result = call_python(request, env)

    scored =
      Enum.map(result["scores"], fn s ->
        {s["idx"], s["reward"], s["stabilized"] || s["landings"] || 0}
      end)

    {scored, result["baseline_reward"], result["baseline_landings"]}
  end

  @doc """
  Collect per-state per-action successor quality via lookahead rollouts.
  Returns {steps_data, n_successes, n_episodes} where steps_data is
  [{state, %{action_atom => rollout_reward}}, ...].
  """
  def get_successor_data(chain, default_action, opts \\ []) do
    env = Keyword.get(opts, :env, :tetris)
    seeds = Keyword.get(opts, :seeds, Enum.to_list(0..9))
    max_steps = Keyword.get(opts, :max_steps, 2000)
    lookahead = Keyword.get(opts, :lookahead, 100)
    sample_every = Keyword.get(opts, :sample_every, 10)

    request = %{
      "cmd" => "successor_explore",
      "chain" => serialize_chain(chain, env),
      "default" => serialize_action(default_action, env),
      "seeds" => seeds,
      "max_steps" => max_steps,
      "lookahead" => lookahead,
      "sample_every" => sample_every
    }

    result = call_python(request, env)

    rev_map = reverse_action_map(env)

    steps =
      Enum.map(result["steps"], fn [state, _chosen, rewards] ->
        atom_rewards =
          rewards
          |> Enum.map(fn {k, v} -> {Map.get(rev_map, String.to_integer(k)), v} end)
          |> Map.new()
        {state, atom_rewards}
      end)

    {steps, result["n_landings"] || 0, result["n_episodes"] || length(seeds)}
  end

  @doc """
  Score candidate predicates locally using pre-collected successor data.
  No Python call -- pure Elixir predicate evaluation + map lookup.
  Returns {scored, baseline_reward, 0} matching score_candidates format.
  """
  def score_candidates_local(candidates, action, default, chain_before, chain_after, succ_data) do
    base_chain = chain_before ++ chain_after

    baseline =
      Enum.reduce(succ_data, 0.0, fn {state, rewards}, acc ->
        chosen = eval_chain(base_chain, default, state)
        acc + Map.get(rewards, chosen, 0.0)
      end)

    scored =
      candidates
      |> Enum.with_index()
      |> Task.async_stream(fn {pred, idx} ->
        chain = chain_before ++ [{pred, action}] ++ chain_after
        total =
          Enum.reduce(succ_data, 0.0, fn {state, rewards}, acc ->
            chosen = eval_chain(chain, default, state)
            acc + Map.get(rewards, chosen, 0.0)
          end)
        {idx, total, 0}
      end, ordered: true, max_concurrency: System.schedulers_online())
      |> Enum.map(fn {:ok, result} -> result end)

    {scored, baseline, 0}
  end

  # ── Learning Logic: Feature Generation ─────────────────────────

  @doc """
  Generate features from raw trajectory states.
  Axis features at percentile thresholds + near-zero band, plus diagonal features.
  """
  def generate_features(states, opts \\ []) do
    env = Keyword.get(opts, :env, :lunarlander)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_dims = num_dims(env)

    axis_feats = generate_axis_features(states, n_dims)
    diag_feats = generate_diag_features(n_dims, max_coeff)
    sq_diag_feats = generate_sq_diag_features(n_dims, max_coeff)
    prod_feats = generate_product_features(states, n_dims)
    tridiag_feats = generate_tridiag_features(n_dims, max_coeff)

    axis_feats ++ diag_feats ++ sq_diag_feats ++ prod_feats ++ tridiag_feats
  end

  defp generate_axis_features(states, n_dims) do
    for dim <- 0..(n_dims - 1),
        t <- axis_thresholds(states, dim) do
      ["axis", dim, t]
    end
  end

  defp axis_thresholds(states, dim) do
    vals = Enum.map(states, fn s -> Enum.at(s, dim) end)
    percentiles = percentile_values(vals, Enum.to_list(0..100//2))
    near_zero = for i <- -15..15, do: i / 100.0

    ([0.0] ++ percentiles ++ near_zero)
    |> Enum.map(&Float.round(&1 * 1.0, 6))
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp generate_diag_features(n_dims, max_coeff) do
    coeffs = for c <- 1..max_coeff, v <- [c, -c], do: v

    for i <- 0..(n_dims - 1),
        j <- 0..(n_dims - 1),
        i != j,
        c <- coeffs do
      ["diag", i, j, c]
    end
  end

  defp generate_sq_diag_features(n_dims, max_coeff) do
    coeffs =
      for c <- [0.01, 0.05, 0.1, 0.2, 0.5] ++ Enum.to_list(1..max_coeff),
          v <- [c, -c], do: v

    for i <- 0..(n_dims - 1),
        j <- 0..(n_dims - 1),
        i != j,
        c <- coeffs do
      ["sq_diag", i, j, c]
    end
  end

  defp generate_product_features(_states, n_dims) when n_dims < 2, do: []
  defp generate_product_features(states, n_dims) do
    pairs = for i <- 0..(n_dims - 2), j <- (i + 1)..(n_dims - 1), do: {i, j}

    Enum.flat_map(pairs, fn {i, j} ->
      products = Enum.map(states, fn s -> Enum.at(s, i) * Enum.at(s, j) end)
      thresholds = percentile_values(products, Enum.to_list(0..100//5))
      near_zero = for k <- -10..10, do: k / 20.0

      ts =
        ([0.0] ++ thresholds ++ near_zero)
        |> Enum.map(&Float.round(&1 * 1.0, 6))
        |> Enum.uniq()
        |> Enum.sort()

      Enum.map(ts, fn t -> ["prod", i, j, t] end)
    end)
  end

  defp generate_tridiag_features(n_dims, _max_coeff) when n_dims < 3, do: []
  defp generate_tridiag_features(n_dims, max_coeff) do
    coeffs = for c <- 1..max_coeff, v <- [c, -c], do: v

    for i <- 0..(n_dims - 1),
        j <- 0..(n_dims - 1),
        k <- 0..(n_dims - 1),
        i != j and j != k and i != k,
        c1 <- coeffs,
        c2 <- coeffs do
      ["tridiag", i, j, k, c1, c2]
    end
  end

  # ── Learning Logic: CEGAR Feature Refinement ──────────────────

  @doc """
  Generic CEGAR feature refinement: run the current policy, collect
  trajectory states, generate new features from regions the policy visits.
  Works for any environment.

  Returns {new_features, n_successes, n_failures}.
  """
  def refine_features(chain, default_action, existing_features, opts \\ []) do
    env = Keyword.get(opts, :env, :lunarlander)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_dims = num_dims(env)
    seeds = Keyword.get(opts, :cegar_seeds, Enum.to_list(0..49))

    {states, n_succ} = get_trajectory_states(chain, default_action,
      env: env, seeds: seeds, max_steps: Keyword.get(opts, :max_steps, 300))

    existing_set = MapSet.new(existing_features, fn f -> List.to_tuple(f) end)

    new_axis = generate_axis_features(states, n_dims)
      |> Enum.reject(fn f -> MapSet.member?(existing_set, List.to_tuple(f)) end)

    new_diag = generate_diag_features(n_dims, max_coeff)
      |> Enum.reject(fn f -> MapSet.member?(existing_set, List.to_tuple(f)) end)

    new_features = Enum.uniq(new_axis ++ new_diag)
    {new_features, n_succ, length(seeds) - n_succ}
  end

  @doc """
  StateCEGAR for environments with multi-step lookahead (explore).
  Uses per-state regret to find counterexamples, generates targeted features.

  Returns {new_features, cex_data, n_cex, n_successes, n_failures}.
  """
  def find_counterexamples(chain, default_action, existing_features, opts \\ []) do
    env = Keyword.get(opts, :env, :pendulum)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_dims = num_dims(env)
    regret_threshold = Keyword.get(opts, :regret_threshold, 0.5)
    max_cex = Keyword.get(opts, :max_cex, 100)

    {steps_data, n_stab, n_episodes} = explore(chain, default_action, opts)

    cex_with_regret =
      steps_data
      |> Enum.map(fn [state, chosen, rewards_map] ->
        chosen_reward = Map.get(rewards_map, to_string(chosen), 0.0)
        best_reward = rewards_map |> Map.values() |> Enum.max()
        {state, best_reward - chosen_reward}
      end)
      |> Enum.filter(fn {_state, regret} -> regret > regret_threshold end)
      |> Enum.sort_by(fn {_state, regret} -> -regret end)

    cex_states = cex_with_regret |> Enum.take(max_cex) |> Enum.map(fn {s, _} -> s end)

    existing_set = MapSet.new(existing_features, fn f -> List.to_tuple(f) end)

    cex_axis =
      for dim <- 0..(n_dims - 1),
          state <- cex_states,
          offset <- [-0.02, -0.01, 0.0, 0.01, 0.02],
          t = Float.round(Enum.at(state, dim) + offset, 6),
          feat = ["axis", dim, t],
          not MapSet.member?(existing_set, List.to_tuple(feat)) do
        feat
      end
      |> Enum.uniq()

    diag_new =
      generate_diag_features(n_dims, max_coeff)
      |> Enum.reject(fn f -> MapSet.member?(existing_set, List.to_tuple(f)) end)

    new_features = Enum.uniq(cex_axis ++ diag_new)

    cex_data = extract_cex_data(steps_data, env, max_cex: max_cex, regret_threshold: regret_threshold)

    IO.puts("  Top regrets: #{inspect(Enum.take(cex_with_regret, 5) |> Enum.map(fn {_, r} -> Float.round(r, 1) end))}")

    {new_features, cex_data, length(cex_with_regret), n_stab, n_episodes - n_stab}
  end

  # ── Local predicate evaluation (pure Elixir, no Python) ─────

  def eval_pred(:truep, _state), do: true
  def eval_pred(:falsep, _state), do: false
  def eval_pred({:feat, ["axis", dim, t]}, state), do: Enum.at(state, dim) < t
  def eval_pred({:feat, ["diag", i, j, c]}, state), do: c * Enum.at(state, i) + Enum.at(state, j) < 0
  def eval_pred({:feat, ["sq_diag", i, j, c]}, state), do: c * Enum.at(state, i) * Enum.at(state, i) + Enum.at(state, j) < 0
  def eval_pred({:feat, ["prod", i, j, t]}, state), do: Enum.at(state, i) * Enum.at(state, j) < t
  def eval_pred({:not, p}, state), do: not eval_pred(p, state)
  def eval_pred({:and, p, q}, state), do: eval_pred(p, state) and eval_pred(q, state)
  def eval_pred({:or, p, q}, state), do: eval_pred(p, state) or eval_pred(q, state)

  def eval_chain([], default, _state), do: default
  def eval_chain([{pred, action} | rest], default, state) do
    if eval_pred(pred, state), do: action, else: eval_chain(rest, default, state)
  end

  # ── Hard consistency filter ──────────────────────────────────

  @doc """
  Filter candidates by consistency with the oracle at CEX states.
  """
  def filter_viable(candidates, action, before, after_chain, default, cex_data, threshold \\ 1.0) do
    n_cex = length(cex_data)
    min_ok = ceil(threshold * n_cex)

    Enum.filter(candidates, fn pred ->
      chain = before ++ [{pred, action}] ++ after_chain
      n_ok = Enum.count(cex_data, fn {state, rewards_map} ->
        chosen = eval_chain(chain, default, state)
        {best_action, _} = Enum.max_by(rewards_map, fn {_a, r} -> r end)
        chosen == best_action
      end)
      n_ok >= min_ok
    end)
  end

  @doc """
  Progressive relaxation: try strict consistency first, then relax.
  """
  def filter_viable_relaxed(candidates, action, before, after_chain, default, cex_data) do
    thresholds = [1.0, 0.9, 0.8, 0.7, 0.5]

    result = Enum.reduce_while(thresholds, nil, fn threshold, _acc ->
      viable = filter_viable(candidates, action, before, after_chain, default, cex_data, threshold)
      if length(viable) > 0 do
        pct = round(threshold * 100)
        IO.puts("    CEX filter: #{length(candidates)} -> #{length(viable)} viable (#{pct}% consistency)")
        {:halt, viable}
      else
        {:cont, nil}
      end
    end)

    case result do
      nil ->
        IO.puts("    CEX filter: no viable at any threshold, using all #{length(candidates)}")
        candidates
      viable -> viable
    end
  end

  def reverse_action_map(env) do
    action_map(env) |> Enum.map(fn {k, v} -> {v, k} end) |> Map.new()
  end

  defp extract_cex_data(steps_data, env, opts) do
    rev_map = reverse_action_map(env)
    max_cex = Keyword.get(opts, :max_cex, 100)
    threshold = Keyword.get(opts, :regret_threshold, 0.5)

    steps_data
    |> Enum.map(fn [state, chosen, rewards] ->
      chosen_r = Map.get(rewards, to_string(chosen), 0.0)
      best_r = rewards |> Map.values() |> Enum.max()
      {state, chosen, rewards, best_r - chosen_r}
    end)
    |> Enum.filter(fn {_, _, _, regret} -> regret > threshold end)
    |> Enum.sort_by(fn {_, _, _, regret} -> -regret end)
    |> Enum.take(max_cex)
    |> Enum.map(fn {state, _chosen, rewards, _regret} ->
      atom_rewards =
        rewards
        |> Enum.map(fn {k, v} -> {Map.get(rev_map, String.to_integer(k)), v} end)
        |> Map.new()
      {state, atom_rewards}
    end)
  end

  # ── Pretty-printing ─────────────────────────────────────────

  def dim_name(env, d), do: Map.get((@envs[env] || %{})[:dim_names] || %{}, d, "d#{d}")
  def dim_py(env, d), do: Map.get((@envs[env] || %{})[:dim_py] || %{}, d, "obs[#{d}]")
  def obs_unpack(env), do: (@envs[env] || %{})[:obs_unpack] || "    # observation"

  def format_pred(:truep, _env), do: "true"
  def format_pred(:falsep, _env), do: "false"
  def format_pred({:feat, ["axis", d, t]}, env), do: "#{dim_name(env, d)}<#{t}"
  def format_pred({:feat, ["diag", i, j, c]}, env), do: "#{c}·#{dim_name(env, i)}+#{dim_name(env, j)}<0"
  def format_pred({:not, p}, env), do: "¬(#{format_pred(p, env)})"
  def format_pred({:and, p, q}, env), do: "(#{format_pred(p, env)} ∧ #{format_pred(q, env)})"
  def format_pred({:or, p, q}, env), do: "(#{format_pred(p, env)} ∨ #{format_pred(q, env)})"
  def format_pred(other, _env), do: inspect(other)

  def fmt_py({:feat, ["axis", d, t]}, env), do: "#{dim_py(env, d)} < #{t}"
  def fmt_py({:feat, ["diag", i, j, c]}, env), do: "#{c}*#{dim_py(env, i)} + #{dim_py(env, j)} < 0"
  def fmt_py({:not, p}, env), do: "not (#{fmt_py(p, env)})"
  def fmt_py({:and, p, q}, env), do: "(#{fmt_py(p, env)}) and (#{fmt_py(q, env)})"
  def fmt_py({:or, p, q}, env), do: "(#{fmt_py(p, env)}) or (#{fmt_py(q, env)})"
  def fmt_py(other, _env), do: inspect(other)

  def print_deployable(chain, default, env) do
    IO.puts("\n=== DEPLOYABLE POLICY ===")
    IO.puts("def policy(obs):")
    IO.puts(obs_unpack(env))

    Enum.each(chain, fn {pred, action} ->
      IO.puts("    if #{fmt_py(pred, env)}: return #{inspect(action)}")
    end)

    IO.puts("    return #{inspect(default)}")
  end

  # ── Utilities ──────────────────────────────────────────────────

  defp percentile_values(vals, percentiles) do
    sorted = Enum.sort(vals)
    n = length(sorted)
    if n == 0 do
      []
    else
      Enum.map(percentiles, fn p ->
        idx = min(round(p / 100.0 * (n - 1)), n - 1)
        Enum.at(sorted, idx)
      end)
    end
  end

  # ── Python bridge ──────────────────────────────────────────────

  defp call_python(request, env) do
    script = oracle_script(env)
    python = Application.get_env(:synthex, :python, "python3")
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_req_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_resp_#{uid}.json")

    File.write!(req_file, Jason.encode!(request))

    {_output, exit_code} =
      System.cmd(python, ["-u", script, req_file, resp_file],
        stderr_to_stdout: true,
        cd: project_root()
      )

    if exit_code != 0 do
      IO.puts("  [Gym.Oracle] Python exited with code #{exit_code}")
    end

    result = Jason.decode!(File.read!(resp_file))
    File.rm(req_file)
    File.rm(resp_file)
    result
  end

  # ── Serialization ──────────────────────────────────────────────

  def serialize_pred(:truep), do: "truep"
  def serialize_pred(:falsep), do: "falsep"
  def serialize_pred({:feat, f}) when is_list(f), do: ["feat", f]
  def serialize_pred({:feat, {:axis, d, t}}), do: ["feat", ["axis", d, t]]
  def serialize_pred({:feat, {:diag, i, j, c}}), do: ["feat", ["diag", i, j, c]]
  def serialize_pred({:not, p}), do: ["not", serialize_pred(p)]
  def serialize_pred({:and, p, q}), do: ["and", serialize_pred(p), serialize_pred(q)]
  def serialize_pred({:or, p, q}), do: ["or", serialize_pred(p), serialize_pred(q)]

  def serialize_action(a, env \\ :lunarlander) do
    Map.fetch!(action_map(env), a)
  end

  def serialize_chain(chain, env \\ :lunarlander) do
    Enum.map(chain, fn {pred, action} ->
      [serialize_pred(pred), serialize_action(action, env)]
    end)
  end
end

# Backward-compatible alias
defmodule Synthex.GymOracle do
  @moduledoc false
  defdelegate action_map(env), to: Synthex.Gym.Oracle
  defdelegate oracle_script(env), to: Synthex.Gym.Oracle
  defdelegate num_dims(env), to: Synthex.Gym.Oracle
  defdelegate get_trajectory_states(chain, default, opts), to: Synthex.Gym.Oracle
  defdelegate explore(chain, default, opts), to: Synthex.Gym.Oracle
  defdelegate score_candidates(cands, action, default, chain, opts), to: Synthex.Gym.Oracle
  defdelegate get_successor_data(chain, default, opts), to: Synthex.Gym.Oracle
  defdelegate score_candidates_local(cands, action, default, before, after_chain, data), to: Synthex.Gym.Oracle
  defdelegate generate_features(states, opts), to: Synthex.Gym.Oracle
  defdelegate refine_features(chain, default, features, opts), to: Synthex.Gym.Oracle
  defdelegate find_counterexamples(chain, default, features, opts), to: Synthex.Gym.Oracle
  defdelegate eval_pred(pred, state), to: Synthex.Gym.Oracle
  defdelegate eval_chain(chain, default, state), to: Synthex.Gym.Oracle
  defdelegate filter_viable(cands, action, before, after_chain, default, data, threshold), to: Synthex.Gym.Oracle
  defdelegate filter_viable_relaxed(cands, action, before, after_chain, default, data), to: Synthex.Gym.Oracle
  defdelegate reverse_action_map(env), to: Synthex.Gym.Oracle
  defdelegate format_pred(pred, env), to: Synthex.Gym.Oracle
  defdelegate serialize_pred(pred), to: Synthex.Gym.Oracle
  defdelegate serialize_action(a, env), to: Synthex.Gym.Oracle
  defdelegate serialize_chain(chain, env), to: Synthex.Gym.Oracle
  defdelegate print_deployable(chain, default, env), to: Synthex.Gym.Oracle
end
