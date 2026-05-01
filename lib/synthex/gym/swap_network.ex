defmodule Synthex.Gym.SwapNetwork do
  @moduledoc """
  CSHRL-grounded synthesis for discrete action rankings via sorting networks.

  Instead of searching over n! permutations, the ranking is produced by
  applying a fixed sorting network (sequence of swap positions) where each
  swap is conditionally applied based on a binary predicate over the state.

  Each swap is an independent 2-action CSHRL problem:
    - Actions: {SWAP, NO_SWAP}
    - Predicate P_i partitions state space into 2 regions
    - Region where P_i holds: apply swap(pos_a, pos_b)
    - Region where ¬P_i: don't swap

  For n actions, the bubble sort network has n(n-1)/2 comparators.
  This gives 2^(n(n-1)/2) reachable rankings, covering all n! permutations.

  Coordinate descent optimizes each swap's predicate independently via
  CEGAR + episode reward evaluation.

  Scaling: 4 actions → 6 swaps, 10 actions → 45 swaps (vs 3.6M permutations).
  """

  alias Synthex.Gym.Oracle, as: GymOracle
  alias Synthex.Core.CEGIS

  @env_configs %{
    lunarlander: %{
      env_name: "LunarLander-v3",
      actions: [:do_nothing, :fire_left, :fire_main, :fire_right],
      action_ids: [0, 1, 2, 3],
      n_dims: 6,
      max_steps: 300,
      dim_names: %{0 => "x", 1 => "y", 2 => "vx", 3 => "vy",
                   4 => "angle", 5 => "angvel"}
    },
    cartpole: %{
      env_name: "CartPole-v1",
      actions: [:left, :right],
      action_ids: [0, 1],
      n_dims: 4,
      max_steps: 500,
      dim_names: %{0 => "x", 1 => "xdot", 2 => "theta", 3 => "thetadot"}
    },
    mountaincar: %{
      env_name: "MountainCar-v0",
      actions: [:push_left, :no_push, :push_right],
      action_ids: [0, 1, 2],
      n_dims: 2,
      max_steps: 200,
      dim_names: %{0 => "position", 1 => "velocity"}
    },
    acrobot: %{
      env_name: "Acrobot-v1",
      actions: [:torque_neg, :no_torque, :torque_pos],
      action_ids: [0, 1, 2],
      n_dims: 6,
      max_steps: 500,
      dim_names: %{0 => "cos1", 1 => "sin1", 2 => "cos2", 3 => "sin2",
                   4 => "angvel1", 5 => "angvel2"}
    }
  }

  def bubble_sort_network(n, passes \\ 1) do
    single_pass =
      for pass_idx <- 0..(n - 2),
          i <- 0..(n - 2 - pass_idx) do
        {i, i + 1}
      end
    List.duplicate(single_pass, passes) |> List.flatten()
  end

  def solve(opts \\ []) do
    mode = Keyword.get(opts, :mode, :coord_descent)
    case mode do
      :progressive -> solve_progressive(opts)
      _ -> solve_coord_descent(opts)
    end
  end

  def solve_coord_descent(opts) do
    env = Keyword.get(opts, :env, :lunarlander)
    config = Map.fetch!(@env_configs, env)
    n_actions = length(config.action_ids)
    passes = Keyword.get(opts, :passes, 1)
    network = bubble_sort_network(n_actions, passes)
    n_swaps = length(network)

    depth = Keyword.get(opts, :depth, 1)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_episodes = Keyword.get(opts, :n_episodes, 30)
    top_k = Keyword.get(opts, :top_k, 20)
    max_iters = Keyword.get(opts, :max_iters, 5)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 3)

    val_seeds = Enum.to_list(10_000..10_499)

    IO.puts("══════════════════════════════════════════════════════")
    IO.puts("  CSHRL Swap Network Action Ranking Synthesis")
    IO.puts("  Mode: coordinate descent")
    IO.puts("  Env: #{env} (#{config.env_name})")
    IO.puts("  Actions: #{inspect(config.actions)}")
    IO.puts("  Sorting network: #{n_swaps} swaps for #{n_actions} actions (#{passes} pass#{if passes > 1, do: "es", else: ""})")
    IO.puts("  Network topology: #{inspect(network)}")
    IO.puts("  Depth: #{depth}, Episodes: #{n_episodes}, TopK: #{top_k}")
    IO.puts("  Coord-descent iters: #{max_iters}, CEGAR rounds: #{cegar_rounds}")
    IO.puts("══════════════════════════════════════════════════════\n")

    IO.puts("── Searching best base ranking ──")
    {base_ranking, base_reward, base_succ} =
      search_base_ranking(config, n_episodes, val_seeds)
    IO.puts("  Best base: #{format_ranking(base_ranking, config)}")
    IO.puts("  Reward: #{Float.round(base_reward, 1)}, landings: #{base_succ}/#{length(val_seeds)}\n")

    swap_preds = List.duplicate(:falsep, n_swaps)

    {_last_preds, best} =
      Enum.reduce(1..cegar_rounds, {swap_preds, {swap_preds, base_ranking, -999_999.0, 0}},
        fn cegar_iter, {preds, best_acc} ->
          IO.puts("\n████████ CEGAR Round #{cegar_iter}/#{cegar_rounds} ████████")

          IO.puts("  Collecting states from current policy...")
          {states, n_succ} = collect_states(preds, base_ranking, network, config)
          IO.puts("  #{length(states)} states collected (#{n_succ} successes)")

          IO.puts("  Generating features...")
          features = GymOracle.generate_features(states, env: env, max_coeff: max_coeff)
          IO.puts("  #{length(features)} features\n")

          {new_preds, new_best} =
            run_coord_descent(preds, base_ranking, network, features,
                              depth, top_k, n_episodes, max_iters,
                              cegar_iter, val_seeds, best_acc, config)

          {new_preds, new_best}
        end)

    {best_preds, _best_base, best_val, best_succ} = best

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  SYNTHESIS COMPLETE")
    IO.puts("  Best validation: #{Float.round(best_val, 1)} (#{best_succ}/#{length(val_seeds)} landings)")
    IO.puts("══════════════════════════════════════════════════════")

    print_policy(best_preds, base_ranking, network, config)
    validate_and_print(best_preds, base_ranking, network, val_seeds, config)
    {base_ranking, best_preds}
  end

  # ── Progressive mode: learn swaps sequentially, discharge outcomes as features ──

  def solve_progressive(opts) do
    env = Keyword.get(opts, :env, :lunarlander)
    config = Map.fetch!(@env_configs, env)
    n_actions = length(config.action_ids)
    passes = Keyword.get(opts, :passes, 1)
    network = bubble_sort_network(n_actions, passes)
    n_swaps = length(network)

    depth = Keyword.get(opts, :depth, 1)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_episodes = Keyword.get(opts, :n_episodes, 30)
    top_k = Keyword.get(opts, :top_k, 20)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 3)

    val_seeds = Enum.to_list(10_000..10_499)

    IO.puts("══════════════════════════════════════════════════════")
    IO.puts("  CSHRL Swap Network — Progressive Mode")
    IO.puts("  Env: #{env} (#{config.env_name})")
    IO.puts("  Actions: #{inspect(config.actions)}")
    IO.puts("  Sorting network: #{n_swaps} swaps (#{passes} pass#{if passes > 1, do: "es", else: ""})")
    IO.puts("  Topology: #{inspect(network)}")
    IO.puts("  Depth: #{depth}, Episodes: #{n_episodes}, TopK: #{top_k}")
    IO.puts("  CEGAR rounds: #{cegar_rounds}")
    IO.puts("  Discharged swap-outcome features: ON")
    IO.puts("══════════════════════════════════════════════════════\n")

    IO.puts("── Searching best base ranking ──")
    {base_ranking, base_reward, base_succ} =
      search_base_ranking(config, n_episodes, val_seeds)
    IO.puts("  Best base: #{format_ranking(base_ranking, config)}")
    IO.puts("  Reward: #{Float.round(base_reward, 1)}, landings: #{base_succ}/#{length(val_seeds)}\n")

    swap_preds = List.duplicate(:falsep, n_swaps)

    {_last_preds, best} =
      Enum.reduce(1..cegar_rounds, {swap_preds, {swap_preds, base_ranking, -999_999.0, 0}},
        fn cegar_iter, {preds, best_acc} ->
          IO.puts("\n████████ CEGAR Round #{cegar_iter}/#{cegar_rounds} ████████")

          IO.puts("  Collecting states from current policy...")
          {states, n_succ} = collect_states(preds, base_ranking, network, config)
          IO.puts("  #{length(states)} states collected (#{n_succ} successes)")

          IO.puts("  Generating features...")
          base_features = GymOracle.generate_features(states, env: env, max_coeff: max_coeff)
          IO.puts("  #{length(base_features)} base features\n")

          {new_preds, new_best} =
            run_progressive_pass(preds, base_ranking, network, base_features,
                                 depth, top_k, n_episodes,
                                 cegar_iter, val_seeds, best_acc, config)

          {new_preds, new_best}
        end)

    {best_preds, _best_base, best_val, best_succ} = best

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  SYNTHESIS COMPLETE")
    IO.puts("  Best validation: #{Float.round(best_val, 1)} (#{best_succ}/#{length(val_seeds)} landings)")
    IO.puts("══════════════════════════════════════════════════════")

    print_policy(best_preds, base_ranking, network, config)
    validate_and_print(best_preds, base_ranking, network, val_seeds, config)
    {base_ranking, best_preds}
  end

  defp run_progressive_pass(preds, base_ranking, network, base_features,
                            depth, top_k, n_episodes,
                            cegar_iter, val_seeds, best_so_far, config) do
    n_swaps = length(network)
    seed_offset = (cegar_iter - 1) * n_swaps * n_episodes
    locked_preds = List.duplicate(nil, n_swaps)

    {final_preds, final_best} =
      Enum.reduce(0..(n_swaps - 1), {preds, best_so_far}, fn swap_idx, {cur_preds, best} ->
        {pos_a, pos_b} = Enum.at(network, swap_idx)
        seeds_start = seed_offset + swap_idx * n_episodes
        seeds = Enum.to_list(seeds_start..(seeds_start + n_episodes - 1))

        discharged = Enum.take(cur_preds, swap_idx)
        n_discharged = length(Enum.reject(discharged, fn p -> p == :falsep end))

        IO.puts("\n  >> Swap #{swap_idx}/#{n_swaps - 1}: positions (#{pos_a},#{pos_b}) [#{n_discharged} discharged features]")

        features_with_discharged =
          base_features ++ build_discharged_features(cur_preds, swap_idx)

        if length(features_with_discharged) > length(base_features) do
          IO.puts("    Features: #{length(base_features)} base + #{length(features_with_discharged) - length(base_features)} discharged = #{length(features_with_discharged)}")
        end

        result = optimize_swap(cur_preds, swap_idx, base_ranking, network,
                               features_with_discharged, depth, top_k, seeds, config)

        new_preds = case result do
          nil ->
            IO.puts("    No improvement (locked as falsep)")
            cur_preds
          {new_pred, reward} ->
            IO.puts("    ✓ #{format_pred(new_pred, config)}  reward=#{Float.round(reward, 1)}")
            List.replace_at(cur_preds, swap_idx, new_pred)
        end

        {val_reward, val_succ} =
          validate_swaps(new_preds, base_ranking, network, val_seeds, config)
        IO.puts("    ▸ Validation: reward=#{Float.round(val_reward, 1)} landings=#{val_succ}/#{length(val_seeds)}")

        {_prev_preds, _prev_base, prev_val, _prev_succ} = best
        new_best = if val_reward > prev_val do
          IO.puts("    ★ New best!")
          {new_preds, base_ranking, val_reward, val_succ}
        else
          best
        end

        {new_preds, new_best}
      end)

    {final_preds, final_best}
  end

  defp build_discharged_features(preds, up_to_idx) do
    preds
    |> Enum.take(up_to_idx)
    |> Enum.with_index()
    |> Enum.reject(fn {p, _} -> p == :falsep end)
    |> Enum.flat_map(fn {pred, idx} ->
      serialized = GymOracle.serialize_pred(pred)
      [
        ["swap_outcome", idx, serialized],
        ["swap_outcome_neg", idx, serialized]
      ]
    end)
  end

  # ── Base ranking search ──────────────────────────────────────────

  defp search_base_ranking(config, _n_episodes, val_seeds) do
    request = %{
      "cmd" => "search_base",
      "env_name" => config.env_name,
      "n_actions" => length(config.action_ids),
      "n_dims" => config.n_dims,
      "seeds" => val_seeds,
      "max_steps" => config.max_steps
    }

    result = call_python(request)
    {result["best_ranking"], result["reward"], result["landings"]}
  end

  # ── Coordinate descent ──────────────────────────────────────────

  defp run_coord_descent(preds, base_ranking, network, features,
                         depth, top_k, n_episodes, max_iters,
                         cegar_iter, val_seeds, best_so_far, config) do
    n_swaps = length(network)

    Enum.reduce(1..max_iters, {preds, best_so_far}, fn iter, {cur_preds, best} ->
      IO.puts("\n╌╌╌ Iteration #{iter}/#{max_iters} ╌╌╌")
      seed_offset = ((cegar_iter - 1) * max_iters + (iter - 1)) * n_episodes
      seeds = Enum.to_list(seed_offset..(seed_offset + n_episodes - 1))

      {new_preds, any_improved} =
        Enum.reduce(0..(n_swaps - 1), {cur_preds, false}, fn swap_idx, {ps, imp} ->
          {pos_a, pos_b} = Enum.at(network, swap_idx)

          IO.puts("\n  >> Swap #{swap_idx}: positions (#{pos_a},#{pos_b})")

          result = optimize_swap(ps, swap_idx, base_ranking, network,
                                 features, depth, top_k, seeds, config)

          case result do
            nil ->
              IO.puts("    No improvement")
              {ps, imp}

            {new_pred, reward} ->
              IO.puts("    ✓ #{format_pred(new_pred, config)}  reward=#{Float.round(reward, 1)}")
              {List.replace_at(ps, swap_idx, new_pred), true}
          end
        end)

      {val_reward, val_succ} =
        validate_swaps(new_preds, base_ranking, network, val_seeds, config)
      IO.puts("  ▸ Validation: reward=#{Float.round(val_reward, 1)} landings=#{val_succ}/#{length(val_seeds)}")

      {_prev_preds, _prev_base, prev_val, _prev_succ} = best
      new_best = if val_reward > prev_val do
        IO.puts("  ★ New best!")
        {new_preds, base_ranking, val_reward, val_succ}
      else
        best
      end

      if not any_improved, do: IO.puts("  Coordinate descent converged.")
      {new_preds, new_best}
    end)
  end

  defp optimize_swap(preds, swap_idx, base_ranking, network,
                     features, max_depth, top_k, seeds, config) do
    atoms = CEGIS.enumerate(features, 0)
    all_d0 = [:truep, :falsep | atoms]

    IO.puts("    Depth 0: #{length(all_d0)} candidates")

    {scored_d0, baseline} =
      score_swap_candidates(all_d0, preds, swap_idx, base_ranking, network, seeds, config)

    IO.puts("    Baseline: #{Float.round(baseline, 1)}")

    best_d0 = Enum.max_by(scored_d0, fn {_i, r, _l} -> r end, fn -> nil end)

    d0_result =
      case best_d0 do
        nil -> nil
        {idx, reward, _count} ->
          if reward > baseline do
            pred = Enum.at(all_d0, idx)
            IO.puts("    Best d0: #{format_pred(pred, config)} reward=#{Float.round(reward, 1)} (Δ=#{Float.round(reward - baseline, 1)})")
            {pred, reward, scored_d0}
          else
            nil
          end
      end

    if max_depth == 0 do
      case d0_result do
        nil -> nil
        {p, r, _} -> {p, r}
      end
    else
      top_atoms =
        scored_d0
        |> Enum.sort_by(fn {_idx, r, _l} -> -r end)
        |> Enum.take(top_k)
        |> Enum.map(fn {idx, _r, _l} -> Enum.at(all_d0, idx) end)
        |> Enum.reject(fn p -> p == :truep or p == :falsep end)

      negations = Enum.map(top_atoms, fn p -> {:not, p} end)

      d1_candidates =
        (for p <- top_atoms, q <- top_atoms, p != q, do: {:and, p, q}) ++
        (for p <- top_atoms, q <- top_atoms, p != q, do: {:or, p, q}) ++
        (for p <- negations, q <- top_atoms, do: {:and, p, q})

      d1_candidates = Enum.uniq(d1_candidates)
      IO.puts("    Depth 1: #{length(d1_candidates)} candidates")

      {scored_d1, _} =
        score_swap_candidates(d1_candidates, preds, swap_idx, base_ranking, network, seeds, config)

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
            pred = Enum.at(d1_candidates, idx)
            IO.puts("    Best d1: #{format_pred(pred, config)} reward=#{Float.round(reward, 1)}")
            {pred, reward}
          else
            case d0_result do
              nil -> nil
              {p, r, _} -> {p, r}
            end
          end
      end
    end
  end

  # ── Python oracle interface ─────────────────────────────────────

  defp score_swap_candidates(candidates, preds, target_idx, base_ranking,
                             network, seeds, config) do
    serialized_candidates = Enum.map(candidates, &GymOracle.serialize_pred/1)
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)
    serialized_network = Enum.map(network, fn {a, b} -> [a, b] end)

    request = %{
      "cmd" => "score_swap",
      "candidates" => serialized_candidates,
      "swap_predicates" => serialized_preds,
      "target_idx" => target_idx,
      "base_ranking" => base_ranking,
      "network" => serialized_network,
      "env_name" => config.env_name,
      "n_actions" => length(config.action_ids),
      "n_dims" => config.n_dims,
      "seeds" => seeds,
      "max_steps" => config.max_steps
    }

    result = call_python(request)

    scored =
      Enum.map(result["scores"], fn s ->
        {s["idx"], s["reward"], s["landings"]}
      end)

    {scored, result["baseline_reward"]}
  end

  defp collect_states(preds, base_ranking, network, config) do
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)
    serialized_network = Enum.map(network, fn {a, b} -> [a, b] end)

    request = %{
      "cmd" => "collect_states",
      "swap_predicates" => serialized_preds,
      "base_ranking" => base_ranking,
      "network" => serialized_network,
      "env_name" => config.env_name,
      "n_actions" => length(config.action_ids),
      "n_dims" => config.n_dims,
      "seeds" => Enum.to_list(0..79),
      "max_steps" => config.max_steps
    }

    result = call_python(request)
    {result["states"], result["n_landings"]}
  end

  defp validate_swaps(preds, base_ranking, network, seeds, config) do
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)
    serialized_network = Enum.map(network, fn {a, b} -> [a, b] end)

    request = %{
      "cmd" => "validate",
      "swap_predicates" => serialized_preds,
      "base_ranking" => base_ranking,
      "network" => serialized_network,
      "env_name" => config.env_name,
      "n_actions" => length(config.action_ids),
      "n_dims" => config.n_dims,
      "seeds" => seeds,
      "max_steps" => config.max_steps
    }

    result = call_python(request)
    {result["reward"], result["landings"]}
  end

  defp call_python(request) do
    script = Path.expand("../../../scripts/gym_swapnet_oracle.py", __DIR__)
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_swapnet_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_swapnet_resp_#{uid}.json")

    File.write!(req_file, Jason.encode!(request))

    {_output, _exit} =
      System.cmd("python3", ["-u", script, req_file, resp_file],
        stderr_to_stdout: true,
        cd: Path.expand("../../..", __DIR__)
      )

    result = Jason.decode!(File.read!(resp_file))
    File.rm(req_file)
    File.rm(resp_file)
    result
  end

  # ── Pretty-printing ─────────────────────────────────────────────

  defp format_ranking(ranking, config) do
    names = Enum.map(ranking, fn id ->
      Enum.find(config.actions, fn a ->
        Map.get(GymOracle.action_map(config_to_env(config)), a) == id
      end) || "action_#{id}"
    end)
    "[#{Enum.join(names, " > ")}]"
  end

  defp config_to_env(config) do
    Enum.find_value(@env_configs, fn {env, c} ->
      if c.env_name == config.env_name, do: env
    end)
  end

  defp print_policy(preds, base_ranking, network, config) do
    IO.puts("\n── Synthesized Swap Network Policy ──")
    IO.puts("  Base ranking: #{format_ranking(base_ranking, config)}")
    IO.puts("  Sorting network: #{length(network)} swaps\n")

    for {swap_idx, {{pos_a, pos_b}, pred}} <-
        Enum.with_index(Enum.zip(network, preds)) |> Enum.map(fn {{s, p}, i} -> {i, {s, p}} end) do
      status = if pred == :falsep, do: "OFF", else: "ON"
      IO.puts("  Swap #{swap_idx} (#{pos_a},#{pos_b}) [#{status}]: #{format_pred(pred, config)}")
      if pred != :falsep do
        IO.puts("    P_#{swap_idx} → SWAP positions #{pos_a}↔#{pos_b}")
        IO.puts("    ¬P_#{swap_idx} → NO SWAP")
      end
    end
    IO.puts("")
  end

  defp validate_and_print(preds, base_ranking, network, seeds, config) do
    {val_reward, val_succ} = validate_swaps(preds, base_ranking, network, seeds, config)
    n = length(seeds)
    avg = Float.round(val_reward / n, 1)

    IO.puts("\n── Final Validation (#{n} seeds) ──")
    IO.puts("  Total reward: #{Float.round(val_reward, 1)}")
    IO.puts("  Average: #{avg}/ep")
    IO.puts("  Successes: #{val_succ}/#{n}")

    IO.puts("\n── For Agda Verification ──")
    IO.puts("  Each swap is an independent 2-action CSHRL problem:")
    for {swap_idx, {{pos_a, pos_b}, pred}} <-
        Enum.with_index(Enum.zip(network, preds)) |> Enum.map(fn {{s, p}, i} -> {i, {s, p}} end) do
      IO.puts("  swap#{swap_idx} (#{pos_a},#{pos_b}): #{format_pred(pred, config)}")
      IO.puts("    P → [SWAP, NO_SWAP]")
      IO.puts("    ¬P → [NO_SWAP, SWAP]")
    end
  end

  defp format_pred(:truep, _config), do: "true"
  defp format_pred(:falsep, _config), do: "false"
  defp format_pred({:feat, ["axis", d, t]}, config),
    do: "#{config.dim_names[d] || "d#{d}"}<#{t}"
  defp format_pred({:feat, ["diag", i, j, c]}, config),
    do: "#{c}·#{config.dim_names[i] || "d#{i}"}+#{config.dim_names[j] || "d#{j}"}<0"
  defp format_pred({:feat, ["sq_diag", i, j, c]}, config),
    do: "#{c}·#{config.dim_names[i] || "d#{i}"}²+#{config.dim_names[j] || "d#{j}"}<0"
  defp format_pred({:feat, ["prod", i, j, t]}, config),
    do: "#{config.dim_names[i] || "d#{i}"}·#{config.dim_names[j] || "d#{j}"}<#{t}"
  defp format_pred({:not, p}, config), do: "¬(#{format_pred(p, config)})"
  defp format_pred({:and, p, q}, config),
    do: "(#{format_pred(p, config)} ∧ #{format_pred(q, config)})"
  defp format_pred({:or, p, q}, config),
    do: "(#{format_pred(p, config)} ∨ #{format_pred(q, config)})"
  defp format_pred(other, _config), do: inspect(other)
end
