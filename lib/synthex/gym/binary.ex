defmodule Synthex.Gym.Binary do
  @moduledoc """
  CSHRL-grounded synthesis for continuous action spaces via binary-weighted
  decomposition.

  Each action dimension is decomposed into k bits with weights {1, 2, 4, ...}.
  Each bit is an independent 2-action CSHRL problem:
    - Actions: {ON, OFF}
    - Predicate P_i partitions state space into 2 regions
    - Region where P_i holds: ranking [ON, OFF]
    - Region where ¬P_i: ranking [OFF, ON]

  The composite continuous action is:
    action_d = 2 * Σ(weight_i * bit_i) / max_sum - 1

  Coordinate descent optimizes each bit's predicate independently via
  CEGAR + episode reward evaluation, using Synthex feature generation
  and the standard GymOracle infrastructure.

  This is a factored CSHRL approach: the product of k independent
  binary CoindHomo-preserving policies.
  """

  alias Synthex.Gym.Oracle, as: GymOracle
  alias Synthex.Core.CEGIS

  @bits_per_dim 3
  @n_action_dims 4
  @n_bits @bits_per_dim * @n_action_dims

  @action_dim_names %{0 => "hip1", 1 => "knee1", 2 => "hip2", 3 => "knee2"}
  @weights for i <- 0..(@bits_per_dim - 1), do: Integer.pow(2, i)
  @max_sum Enum.sum(@weights)

  @bipedal_dims %{
    0 => "hull_angle", 1 => "hull_angvel", 2 => "vel_x", 3 => "vel_y",
    4 => "hip1_angle", 5 => "hip1_speed", 6 => "knee1_angle", 7 => "knee1_speed",
    8 => "leg1_contact",
    9 => "hip2_angle", 10 => "hip2_speed", 11 => "knee2_angle", 12 => "knee2_speed",
    13 => "leg2_contact",
    14 => "lidar0", 15 => "lidar1", 16 => "lidar2", 17 => "lidar3", 18 => "lidar4",
    19 => "lidar5", 20 => "lidar6", 21 => "lidar7", 22 => "lidar8", 23 => "lidar9"
  }

  def solve(opts \\ []) do
    env = Keyword.get(opts, :env, :bipedal)
    max_steps = Keyword.get(opts, :max_steps, 1600)
    depth = Keyword.get(opts, :depth, 1)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_episodes = Keyword.get(opts, :n_episodes, 30)
    top_k = Keyword.get(opts, :top_k, 20)
    max_iters = Keyword.get(opts, :max_iters, 5)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 3)

    val_seeds = Enum.to_list(10_000..10_199)

    IO.puts("══════════════════════════════════════════════════════")
    IO.puts("  CSHRL Binary-Weighted Continuous Action Synthesis")
    IO.puts("  Env: #{env}")
    IO.puts("  #{@bits_per_dim} bits/dim × #{@n_action_dims} dims = #{@n_bits} predicates")
    IO.puts("  #{Integer.pow(2, @bits_per_dim)} levels per dimension in [-1, 1]")
    IO.puts("  Weights: #{inspect(@weights)}, max sum: #{@max_sum}")
    IO.puts("  Depth: #{depth}, Episodes: #{n_episodes}, TopK: #{top_k}")
    IO.puts("  Coord-descent iters: #{max_iters}, CEGAR rounds: #{cegar_rounds}")
    IO.puts("  max_steps: #{max_steps}")
    IO.puts("══════════════════════════════════════════════════════\n")

    IO.puts("Action levels per dimension:")
    for s <- 0..@max_sum do
      val = 2.0 * s / @max_sum - 1.0
      bits = for i <- 0..(@bits_per_dim - 1), do: if(Bitwise.band(Bitwise.bsr(s, i), 1) == 1, do: 1, else: 0)
      IO.puts("  #{inspect(bits)} → #{Float.round(val, 3)}")
    end
    IO.puts("")

    bit_preds = List.duplicate(:falsep, @n_bits)

    {final_preds, _best} =
      Enum.reduce(1..cegar_rounds, {bit_preds, {bit_preds, -999_999.0, 0}},
        fn cegar_iter, {preds, best} ->
          IO.puts("\n████████ CEGAR Round #{cegar_iter}/#{cegar_rounds} ████████")

          IO.puts("  Collecting states from current policy...")
          {states, _n_survived} = collect_states_bits(preds, env, max_steps)
          IO.puts("  #{length(states)} states collected")

          IO.puts("  Generating features in Synthex...")
          features = generate_bipedal_features(states, max_coeff)
          IO.puts("  #{length(features)} features\n")

          {new_preds, new_best} =
            run_coord_descent(preds, features, depth, top_k, n_episodes,
                              max_iters, cegar_iter, val_seeds, best, env, max_steps)

          if cegar_iter < cegar_rounds do
            IO.puts("\n  ─── CEGAR: refining abstraction ───")
            {new_states, n_survived} = collect_states_bits(new_preds, env, max_steps)
            new_feats = GymOracle.generate_features(new_states, env: env, max_coeff: max_coeff)
            n_new = length(new_feats) - length(features)
            IO.puts("  Policy survived: #{n_survived} episodes")
            IO.puts("  New features from on-policy states: #{max(n_new, 0)}")
          end

          {new_preds, new_best}
        end)

    {best_preds, best_val, best_survived} = _best =
      Enum.reduce(1..cegar_rounds, {bit_preds, -999_999.0, 0},
        fn _, acc -> acc end)

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  SYNTHESIS COMPLETE")
    IO.puts("══════════════════════════════════════════════════════")

    print_policy(final_preds, env)
    validate_and_print(final_preds, val_seeds, env, max_steps)
    final_preds
  end

  # ── Coordinate descent ──────────────────────────────────────────

  defp run_coord_descent(preds, features, depth, top_k, n_episodes,
                         max_iters, cegar_iter, val_seeds, best_so_far, env, max_steps) do
    Enum.reduce(1..max_iters, {preds, best_so_far}, fn iter, {cur_preds, best} ->
      IO.puts("\n╌╌╌ Iteration #{iter}/#{max_iters} ╌╌╌")
      seed_offset = ((cegar_iter - 1) * max_iters + (iter - 1)) * n_episodes
      seeds = Enum.to_list(seed_offset..(seed_offset + n_episodes - 1))

      {new_preds, any_improved} =
        Enum.reduce(0..(@n_bits - 1), {cur_preds, false}, fn bit_idx, {ps, imp} ->
          dim_idx = div(bit_idx, @bits_per_dim)
          bit_pos = rem(bit_idx, @bits_per_dim)
          weight = Enum.at(@weights, bit_pos)
          dim_name = @action_dim_names[dim_idx]

          IO.puts("\n  >> Bit #{bit_idx}: #{dim_name} weight=#{weight}")

          result = optimize_bit(ps, bit_idx, features, depth, top_k, seeds, env, max_steps)

          case result do
            nil ->
              IO.puts("    No improvement")
              {ps, imp}

            {new_pred, reward} ->
              IO.puts("    ✓ #{format_pred(new_pred, env)}  reward=#{Float.round(reward, 1)}")
              {List.replace_at(ps, bit_idx, new_pred), true}
          end
        end)

      {val_reward, val_survived} = validate_bits(new_preds, val_seeds, env, max_steps)
      IO.puts("  ▸ Validation: reward=#{Float.round(val_reward, 1)} survived=#{val_survived}/#{length(val_seeds)}")

      {_prev_preds, prev_val, _prev_surv} = best
      new_best = if val_reward > prev_val do
        IO.puts("  ★ New best!")
        {new_preds, val_reward, val_survived}
      else
        best
      end

      if not any_improved, do: IO.puts("  Coordinate descent converged.")
      {new_preds, new_best}
    end)
  end

  defp optimize_bit(preds, bit_idx, features, max_depth, top_k, seeds, env, max_steps) do
    atoms = CEGIS.enumerate(features, 0)

    all_d0 = [:truep, :falsep | atoms]

    IO.puts("    Depth 0: #{length(all_d0)} candidates")

    {scored_d0, baseline} = score_bit_candidates(all_d0, preds, bit_idx, seeds, env, max_steps)

    IO.puts("    Baseline: #{Float.round(baseline, 1)}")

    best_d0 = Enum.max_by(scored_d0, fn {_i, r, _l} -> r end, fn -> nil end)

    d0_result =
      case best_d0 do
        nil -> nil
        {idx, reward, _count} ->
          if reward > baseline do
            pred = Enum.at(all_d0, idx)
            IO.puts("    Best d0: #{format_pred(pred, env)} reward=#{Float.round(reward, 1)} (Δ=#{Float.round(reward - baseline, 1)})")
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

      {scored_d1, _} = score_bit_candidates(d1_candidates, preds, bit_idx, seeds, env, max_steps)

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
            IO.puts("    Best d1: #{format_pred(pred, env)} reward=#{Float.round(reward, 1)}")
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

  defp score_bit_candidates(candidates, preds, target_bit, seeds, env, max_steps) do
    serialized_candidates = Enum.map(candidates, &GymOracle.serialize_pred/1)
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)

    request = %{
      "cmd" => "score_bit",
      "candidates" => serialized_candidates,
      "bit_predicates" => serialized_preds,
      "target_bit" => target_bit,
      "seeds" => seeds,
      "max_steps" => max_steps
    }

    result = call_python(request, env)

    scored =
      Enum.map(result["scores"], fn s ->
        {s["idx"], s["reward"], s["landings"]}
      end)

    {scored, result["baseline_reward"]}
  end

  defp collect_states_bits(preds, env, max_steps) do
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)

    request = %{
      "cmd" => "collect_states",
      "bit_predicates" => serialized_preds,
      "seeds" => Enum.to_list(0..39),
      "max_steps" => max_steps
    }

    result = call_python(request, env)
    {result["states"], result["n_landings"]}
  end

  defp validate_bits(preds, seeds, env, max_steps) do
    serialized_preds = Enum.map(preds, &GymOracle.serialize_pred/1)

    request = %{
      "cmd" => "score_bit",
      "candidates" => [],
      "bit_predicates" => serialized_preds,
      "target_bit" => 0,
      "seeds" => seeds,
      "max_steps" => max_steps
    }

    result = call_python(request, env)
    {result["baseline_reward"], result["baseline_landings"]}
  end

  defp call_python(request, env) do
    script = GymOracle.oracle_script(env)
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_binary_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_binary_resp_#{uid}.json")

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

  # ── Feature generation (limited for 24D) ────────────────────────

  @key_dims [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]

  defp generate_bipedal_features(states, max_coeff) do
    n_dims = 24

    axis_feats = generate_axis_feats(states, n_dims)

    coeffs = for c <- 1..max_coeff, v <- [c, -c], do: v
    diag_feats =
      for i <- @key_dims, j <- @key_dims, i != j, c <- coeffs do
        ["diag", i, j, c]
      end

    sq_coeffs = for c <- [1, 2, 3], v <- [c, -c], do: v
    sq_diag_feats =
      for i <- @key_dims, j <- @key_dims, i != j, c <- sq_coeffs do
        ["sq_diag", i, j, c]
      end

    lidar_physics =
      for ld <- [14, 16, 18, 20, 22],
          pd <- [0, 2, 3],
          c <- [1, -1] do
        ["diag", pd, ld, c]
      end

    all = axis_feats ++ diag_feats ++ sq_diag_feats ++ lidar_physics
    Enum.uniq(all)
  end

  defp generate_axis_feats(states, n_dims) do
    Enum.flat_map(0..(n_dims - 1), fn dim ->
      vals = Enum.map(states, fn s -> Enum.at(s, dim) end)
      pcts = percentile_values(vals, Enum.to_list(0..100//5))
      near_zero = for i <- -10..10, do: i / 100.0

      thresholds =
        ([0.0] ++ pcts ++ near_zero)
        |> Enum.map(&Float.round(&1 * 1.0, 6))
        |> Enum.uniq()
        |> Enum.sort()

      Enum.map(thresholds, fn t -> ["axis", dim, t] end)
    end)
  end

  defp percentile_values(vals, percentiles) do
    sorted = Enum.sort(vals)
    n = length(sorted)
    if n == 0, do: [], else:
      Enum.map(percentiles, fn p ->
        idx = min(round(p / 100.0 * (n - 1)), n - 1)
        Enum.at(sorted, idx)
      end)
  end

  # ── Pretty-printing ─────────────────────────────────────────────

  defp print_policy(preds, env) do
    IO.puts("\n── Synthesized Policy ──")
    for d <- 0..(@n_action_dims - 1) do
      dim_name = @action_dim_names[d]
      IO.puts("\n  #{dim_name}:")
      for i <- 0..(@bits_per_dim - 1) do
        bit_idx = d * @bits_per_dim + i
        p = Enum.at(preds, bit_idx)
        weight = Enum.at(@weights, i)
        IO.puts("    bit#{i} (weight #{weight}): #{format_pred(p, env)}")
      end
    end
    IO.puts("")
  end

  defp validate_and_print(preds, seeds, env, max_steps) do
    {val_reward, val_survived} = validate_bits(preds, seeds, env, max_steps)
    n = length(seeds)
    avg = Float.round(val_reward / n, 1)

    IO.puts("\n── Final Validation (#{n} seeds) ──")
    IO.puts("  Total reward: #{Float.round(val_reward, 1)}")
    IO.puts("  Average: #{avg}/ep")
    IO.puts("  Survived (reward > 0): #{val_survived}/#{n}")

    IO.puts("\n── Deployable Policy ──")
    IO.puts("def policy(obs):")
    IO.puts("    bits = [0] * #{@n_bits}")
    for bit_idx <- 0..(@n_bits - 1) do
      p = Enum.at(preds, bit_idx)
      if p != :falsep do
        d = div(bit_idx, @bits_per_dim)
        i = rem(bit_idx, @bits_per_dim)
        weight = Enum.at(@weights, i)
        dim_name = @action_dim_names[d]
        IO.puts("    # #{dim_name} bit#{i} (weight #{weight})")
        IO.puts("    if #{fmt_py(p, env)}: bits[#{bit_idx}] = 1")
      end
    end
    IO.puts("    weights = #{inspect(@weights)}")
    IO.puts("    actions = [0.0] * #{@n_action_dims}")
    IO.puts("    for d in range(#{@n_action_dims}):")
    IO.puts("        s = sum(weights[i] * bits[d*#{@bits_per_dim}+i] for i in range(#{@bits_per_dim}))")
    IO.puts("        actions[d] = 2.0 * s / #{@max_sum} - 1.0")
    IO.puts("    return actions")

    IO.puts("\n── For Agda Verification ──")
    IO.puts("  Each bit is an independent 2-action CSHRL problem:")
    for bit_idx <- 0..(@n_bits - 1) do
      p = Enum.at(preds, bit_idx)
      d = div(bit_idx, @bits_per_dim)
      i = rem(bit_idx, @bits_per_dim)
      weight = Enum.at(@weights, i)
      dim_name = @action_dim_names[d]
      IO.puts("  bit#{bit_idx} (#{dim_name} w#{weight}): #{format_pred(p, env)}")
      IO.puts("    P_#{bit_idx} → [ON, OFF]")
      IO.puts("    ¬P_#{bit_idx} → [OFF, ON]")
    end
  end

  defp format_pred(:truep, _env), do: "true"
  defp format_pred(:falsep, _env), do: "false"
  defp format_pred({:feat, ["axis", d, t]}, _env), do: "#{@bipedal_dims[d] || "d#{d}"}<#{t}"
  defp format_pred({:feat, ["diag", i, j, c]}, _env), do: "#{c}·#{@bipedal_dims[i] || "d#{i}"}+#{@bipedal_dims[j] || "d#{j}"}<0"
  defp format_pred({:feat, ["sq_diag", i, j, c]}, _env), do: "#{c}·#{@bipedal_dims[i] || "d#{i}"}²+#{@bipedal_dims[j] || "d#{j}"}<0"
  defp format_pred({:feat, ["prod", i, j, t]}, _env), do: "#{@bipedal_dims[i] || "d#{i}"}·#{@bipedal_dims[j] || "d#{j}"}<#{t}"
  defp format_pred({:not, p}, env), do: "¬(#{format_pred(p, env)})"
  defp format_pred({:and, p, q}, env), do: "(#{format_pred(p, env)} ∧ #{format_pred(q, env)})"
  defp format_pred({:or, p, q}, env), do: "(#{format_pred(p, env)} ∨ #{format_pred(q, env)})"
  defp format_pred(other, _env), do: inspect(other)

  @bipedal_py %{
    0 => "obs[0]", 1 => "obs[1]", 2 => "obs[2]", 3 => "obs[3]",
    4 => "obs[4]", 5 => "obs[5]", 6 => "obs[6]", 7 => "obs[7]",
    8 => "obs[8]", 9 => "obs[9]", 10 => "obs[10]", 11 => "obs[11]",
    12 => "obs[12]", 13 => "obs[13]", 14 => "obs[14]", 15 => "obs[15]",
    16 => "obs[16]", 17 => "obs[17]", 18 => "obs[18]", 19 => "obs[19]",
    20 => "obs[20]", 21 => "obs[21]", 22 => "obs[22]", 23 => "obs[23]"
  }

  defp fmt_py({:feat, ["axis", d, t]}, _env), do: "#{@bipedal_py[d]} < #{t}"
  defp fmt_py({:feat, ["diag", i, j, c]}, _env), do: "#{c}*#{@bipedal_py[i]} + #{@bipedal_py[j]} < 0"
  defp fmt_py({:feat, ["sq_diag", i, j, c]}, _env), do: "#{c}*#{@bipedal_py[i]}**2 + #{@bipedal_py[j]} < 0"
  defp fmt_py({:feat, ["prod", i, j, t]}, _env), do: "#{@bipedal_py[i]}*#{@bipedal_py[j]} < #{t}"
  defp fmt_py({:not, p}, env), do: "not (#{fmt_py(p, env)})"
  defp fmt_py({:and, p, q}, env), do: "(#{fmt_py(p, env)}) and (#{fmt_py(q, env)})"
  defp fmt_py({:or, p, q}, env), do: "(#{fmt_py(p, env)}) or (#{fmt_py(q, env)})"
  defp fmt_py(other, _env), do: inspect(other)
end
