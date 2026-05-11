defmodule Synthex.Gym.Mujoco do
  @moduledoc """
  Binary-weighted synthesis for MuJoCo continuous-action environments.

  Generalized version of Binary: bits_per_dim, n_action_dims, and
  dim names are runtime parameters rather than compile-time constants.
  Uses the shared mujoco.py oracle adapter.
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

  def solve(env_key, opts \\ []) do
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

    {lo, hi} = cfg.action_range
    val_seeds = Enum.to_list(10_000..10_199)

    IO.puts("  CSHRL Binary-Weighted Synthesis -- MuJoCo")
    IO.puts("  Env: #{cfg.gym_name}")
    IO.puts("  #{bits_per_dim} bits/dim x #{n_action_dims} dims = #{n_bits} predicates")
    IO.puts("  Obs dims: #{cfg.num_dims} | Action range: [#{lo}, #{hi}]")
    IO.puts("  Depth: #{depth}, Episodes: #{n_episodes}, TopK: #{top_k}")
    IO.puts("  Features: #{inspect(feature_types)} (max_coeff=#{max_coeff}, tridiag_max_coeff=#{tridiag_max_coeff})\n")

    ctx = %{
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
      feature_types: feature_types,
      scorer: scorer
    }

    bit_preds = List.duplicate(:falsep, n_bits)

    final_preds =
      Enum.reduce(1..cegar_rounds, bit_preds, fn cegar_iter, preds ->
        IO.puts("\n  CEGAR Round #{cegar_iter}/#{cegar_rounds}")

        {states, _n_succ} = collect_states(preds, ctx)
        IO.puts("  #{length(states)} states collected")

        features =
          GymOracle.generate_features(states,
            env: env_key,
            n_dims: cfg.num_dims,
            max_coeff: max_coeff,
            tridiag_max_coeff: tridiag_max_coeff,
            tridiag_dims: tridiag_dims,
            feature_types: feature_types
          )

        IO.puts("  #{length(features)} features\n")

        Enum.reduce(1..max_iters, preds, fn iter, cur_preds ->
          IO.puts("\n  Iteration #{iter}/#{max_iters}")
          seed_offset = ((cegar_iter - 1) * max_iters + (iter - 1)) * n_episodes
          seeds = Enum.to_list(seed_offset..(seed_offset + n_episodes - 1))

          bit_indices = Enum.to_list(0..(n_bits - 1)) |> Enum.shuffle()

          {new_preds, _any_improved} =
            Enum.reduce(bit_indices, {cur_preds, false}, fn bit_idx, {ps, imp} ->
              dim_idx = div(bit_idx, bits_per_dim)
              bit_pos = rem(bit_idx, bits_per_dim)
              weight = Enum.at(weights, bit_pos)
              dim_name = cfg.action_dim_names[dim_idx] || "dim#{dim_idx}"

              IO.puts("\n  >> Bit #{bit_idx}: #{dim_name} weight=#{weight}")
              result = optimize_bit(ps, bit_idx, features, ctx, seeds)

              case result do
                nil ->
                  IO.puts("    No improvement")
                  {ps, imp}
                {new_pred, reward} ->
                  IO.puts("    reward=#{Float.round(reward, 1)}")
                  updated = List.replace_at(ps, bit_idx, new_pred)

                  # Emit a telemetry event so masters can react to
                  # accepted CEGAR steps without us shipping a
                  # callback-option API surface. Handlers attach via
                  # `:telemetry.attach/4` with event name
                  # `[:synthex, :mujoco, :bit_accepted]`. Measurements
                  # carry the new reward; metadata carries everything
                  # a snapshot publisher needs.
                  :telemetry.execute(
                    [:synthex, :mujoco, :bit_accepted],
                    %{reward: reward},
                    %{
                      env_key: ctx.env_key,
                      env_name: cfg.gym_name,
                      cegar_iter: cegar_iter,
                      iter: iter,
                      bit_idx: bit_idx,
                      bits_per_dim: ctx.bits_per_dim,
                      n_action_dims: ctx.n_action_dims,
                      n_bits: ctx.n_bits,
                      action_range: cfg.action_range,
                      action_dim_names: cfg.action_dim_names,
                      bit_predicates: updated
                    }
                  )

                  {updated, true}
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

  defp optimize_bit(preds, bit_idx, features, ctx, seeds) do
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

  defp collect_states(preds, ctx) do
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

  defp validate(preds, seeds, ctx) do
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

  # Single point of contact with the outside world. The scorer is
  # whatever the caller plugged in via `solve(..., scorer: ...)`;
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
