defmodule Synthex.Gym.Successor do
  @moduledoc """
  Decision-chain synthesis using successor-state quality scoring with
  episode-level validation fallback.

  Implements the CoinductiveHomomorphism criterion: actions are ranked by
  the quality of the states they lead to. Uses a two-phase approach:

  Phase 1 (fast): Pre-collect per-state per-action successor quality via
  lookahead rollouts, score candidates locally in Elixir.

  Phase 2 (accurate): Validate the top candidates with full episode runs.

  When successor scoring provides differentiation (non-trivial rollout
  policy), Phase 1 filters out most candidates, giving a large speedup.
  When it doesn't differentiate (e.g., first iteration with NOOP policy),
  the system falls back to full episode scoring gracefully.
  """

  alias Synthex.Gym.Oracle, as: GymOracle
  alias Synthex.Core.CEGIS

  @doc """
  Synthesize a decision chain using hybrid successor + episode scoring.

  ## Options
    - `:env`            — environment atom (default :tetris)
    - `:depth`          — max boolean depth (default 1)
    - `:max_coeff`      — max diagonal coefficient (default 5)
    - `:n_episodes`     — episodes per candidate scoring (default 10)
    - `:top_k`          — depth-0 atoms kept for depth-1 (default 30)
    - `:max_iters`      — coordinate descent iters per CEGAR round (default 5)
    - `:cegar_rounds`   — max CEGAR rounds (default 3)
    - `:max_steps`      — **required.** Max steps per episode.
    - `:lookahead`      — rollout depth for successor quality (default 200)
    - `:sample_every`   — sample states every N steps (default 10)
    - `:val_episodes`   — episodes for full-episode validation (default 500)
    - `:succ_top_k`     — candidates to keep from successor pre-filter (default 200)
  """
  def solve(action_priority, default_action, opts \\ []) do
    env = Keyword.get(opts, :env, :tetris)
    depth = Keyword.get(opts, :depth, 1)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_episodes = Keyword.get(opts, :n_episodes, 10)
    top_k = Keyword.get(opts, :top_k, 30)
    max_iters = Keyword.get(opts, :max_iters, 5)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 3)
    max_steps = Keyword.fetch!(opts, :max_steps)
    lookahead = Keyword.get(opts, :lookahead, 200)
    sample_every = Keyword.get(opts, :sample_every, 10)
    val_episodes = Keyword.get(opts, :val_episodes, 500)
    succ_top_k = Keyword.get(opts, :succ_top_k, 200)

    val_seeds = Enum.to_list(10_000..(10_000 + val_episodes - 1))
    n_positions = length(action_priority)

    IO.puts("══════════════════════════════════════════════════════")
    IO.puts("  Successor-State Synthesis — CoinductiveHomomorphism")
    IO.puts("  Env: #{env}")
    IO.puts("  Priority: #{inspect(action_priority)} > #{inspect(default_action)}")
    IO.puts("  Depth: #{depth}, TopK: #{top_k}, SuccTopK: #{succ_top_k}")
    IO.puts("  Coord-descent iters: #{max_iters}, CEGAR rounds: #{cegar_rounds}")
    IO.puts("  Lookahead: #{lookahead}, Sample every: #{sample_every}")
    IO.puts("  Episodes: #{n_episodes}, max_steps: #{max_steps}")
    IO.puts("══════════════════════════════════════════════════════\n")

    IO.puts("  Collecting trajectory states from Gymnasium...")
    {states, _} = GymOracle.get_trajectory_states([], default_action,
      env: env, seeds: Enum.to_list(0..39), max_steps: max_steps)
    IO.puts("  #{length(states)} states collected")

    IO.puts("  Generating features in Synthex...")
    features = GymOracle.generate_features(states, env: env, max_coeff: max_coeff)
    IO.puts("  #{length(features)} initial features\n")

    initial_preds = List.duplicate(:falsep, n_positions)
    initial_chain = Enum.zip(initial_preds, action_priority)
    initial_best = {initial_chain, -999_999.0, 0}

    {_final_chain, final_features, {best_chain, best_val, best_count}} =
      Enum.reduce(1..cegar_rounds, {initial_chain, features, initial_best}, fn cegar_iter, {chain, feats, best} ->
        IO.puts("\n████████ CEGAR Round #{cegar_iter}/#{cegar_rounds} — #{length(feats)} features ████████")

        active = Enum.reject(chain, fn {p, _} -> p == :falsep end)

        IO.puts("  Collecting successor data (lookahead=#{lookahead})...")
        succ_seeds = Enum.to_list(0..(n_episodes - 1))
        {succ_data, n_succ, _} = GymOracle.get_successor_data(active, default_action,
          env: env, seeds: succ_seeds, max_steps: max_steps,
          lookahead: lookahead, sample_every: sample_every)
        IO.puts("  #{length(succ_data)} successor samples (#{n_succ}/#{n_episodes} scoring)")

        {new_chain, new_best} =
          run_coord_descent(chain, default_action, feats, depth, top_k, n_episodes,
                            max_iters, cegar_iter, val_seeds, best, env,
                            succ_data, succ_top_k, max_steps, lookahead, sample_every)

        active_new = Enum.reject(new_chain, fn {p, _} -> p == :falsep end)

        if cegar_iter < cegar_rounds and length(active_new) > 0 do
          IO.puts("\n  ─── CEGAR: refining abstraction ───")

          {traj_feats, n_s, n_f} =
            GymOracle.refine_features(active_new, default_action, feats,
              env: env, max_coeff: max_coeff, max_steps: max_steps)

          IO.puts("  Policy: #{n_s} successes, #{n_f} failures")
          new_feats = Enum.uniq(traj_feats)
          IO.puts("  New features: #{length(new_feats)}")

          if length(new_feats) == 0 do
            IO.puts("  No new features — CEGAR converged!")
            {new_chain, feats, new_best}
          else
            expanded = feats ++ new_feats
            IO.puts("  Feature pool: #{length(feats)} → #{length(expanded)}")
            {new_chain, expanded, new_best}
          end
        else
          {new_chain, feats, new_best}
        end
      end)

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  SYNTHESIS COMPLETE")
    IO.puts("  Best validation: reward=#{Float.round(best_val, 1)} successes=#{best_count}/#{length(val_seeds)}")
    IO.puts("  Features explored: #{length(final_features)}")
    IO.puts("══════════════════════════════════════════════════════")

    active_chain = Enum.reject(best_chain, fn {pred, _} -> pred == :falsep end)

    IO.puts("Final chain (#{length(active_chain)} predicates):")
    Enum.each(active_chain, fn {pred, action} ->
      IO.puts("  #{inspect(action)}  when  #{format_pred(pred, env)}")
    end)
    IO.puts("  #{inspect(default_action)}  otherwise\n")

    print_deployable(active_chain, default_action, env)
    {active_chain, default_action}
  end

  # ── Coordinate descent ─────────────────────────────────────────

  defp run_coord_descent(chain, default, features, depth, top_k, n_episodes,
                          max_iters, cegar_iter, val_seeds, best_so_far, env,
                          succ_data, succ_top_k, max_steps, lookahead, sample_every) do
    Enum.reduce(1..max_iters, {chain, best_so_far}, fn iter, {ch, best} ->
      IO.puts("\n╌╌╌ Iteration #{iter}/#{max_iters} ╌╌╌")
      seed_offset = ((cegar_iter - 1) * max_iters + (iter - 1)) * n_episodes
      seeds = Enum.to_list(seed_offset..(seed_offset + n_episodes - 1))

      current_succ = if iter == 1 do
        succ_data
      else
        active = Enum.reject(ch, fn {p, _} -> p == :falsep end)
        IO.puts("  Refreshing successor data...")
        succ_seeds = Enum.to_list(0..(n_episodes - 1))
        {new_data, _, _} = GymOracle.get_successor_data(active, default,
          env: env, seeds: succ_seeds, max_steps: max_steps,
          lookahead: lookahead, sample_every: sample_every)
        IO.puts("  #{length(new_data)} samples refreshed")
        new_data
      end

      {new_chain, converged} = run_iteration(ch, default, features, depth, top_k,
                                              seeds, env, current_succ, succ_top_k, max_steps)

      active = Enum.reject(new_chain, fn {p, _} -> p == :falsep end)
      new_best = if length(active) > 0 do
        {val_reward, val_count} = validate_chain(active, default, val_seeds, env, max_steps)
        IO.puts("  ▸ Validation: reward=#{Float.round(val_reward, 1)} successes=#{val_count}/#{length(val_seeds)}")

        {_prev_chain, prev_val, _prev_count} = best
        if val_reward > prev_val do
          IO.puts("  ★ New best!")
          {new_chain, val_reward, val_count}
        else
          best
        end
      else
        best
      end

      if converged, do: IO.puts("  Coordinate descent converged.")
      {new_chain, new_best}
    end)
  end

  defp validate_chain(chain, default, seeds, env, max_steps) do
    serialized = GymOracle.serialize_chain(chain, env)
    default_int = GymOracle.serialize_action(default, env)

    request = %{
      "cmd" => "score",
      "candidates" => [],
      "stage_action" => 0,
      "default" => default_int,
      "chain_so_far" => serialized,
      "chain_after" => [],
      "seeds" => seeds,
      "max_steps" => max_steps
    }

    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_val_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_val_resp_#{uid}.json")

    File.write!(req_file, Jason.encode!(request))

    script = GymOracle.oracle_script(env)
    {_output, _exit} =
      System.cmd("python3", ["-u", script, req_file, resp_file],
        stderr_to_stdout: true,
        cd: Path.expand("../../..", __DIR__)
      )

    result = Jason.decode!(File.read!(resp_file))
    File.rm(req_file)
    File.rm(resp_file)

    {result["baseline_reward"], result["baseline_landings"]}
  end

  # ── Per-position optimization with two-phase scoring ──────────

  defp run_iteration(chain, default, features, depth, top_k, seeds, env, succ_data, succ_top_k, max_steps) do
    n = length(chain)

    {new_chain, any_improved} =
      Enum.reduce(0..(n - 1), {chain, false}, fn pos, {ch, imp} ->
        {_pred, action} = Enum.at(ch, pos)
        IO.puts("\n  >> Position #{pos + 1}/#{n}: #{inspect(action)}")

        before = Enum.take(ch, pos)
        after_chain = Enum.drop(ch, pos + 1)

        best = optimize_position_hybrid(
          action, default, before, after_chain, features, depth, top_k,
          seeds, env, succ_data, succ_top_k, max_steps
        )

        case best do
          nil ->
            IO.puts("    No improvement")
            {ch, imp}

          {new_pred, reward, count} ->
            IO.puts("    ✓ #{format_pred(new_pred, env)}  reward=#{Float.round(reward, 1)} count=#{count}")
            {List.replace_at(ch, pos, {new_pred, action}), true}
        end
      end)

    {new_chain, not any_improved}
  end

  defp optimize_position_hybrid(action, default, chain_before, chain_after, features,
                                 max_depth, top_k, seeds, env, succ_data, succ_top_k, max_steps) do
    atoms = CEGIS.enumerate(features, 0)
    IO.puts("    Depth 0: #{length(atoms)} candidates")

    {succ_scored, succ_baseline, _} =
      GymOracle.score_candidates_local(atoms, action, default, chain_before, chain_after, succ_data)

    succ_scores = Enum.map(succ_scored, fn {_i, r, _} -> r end)
    succ_range = (Enum.max(succ_scores, fn -> 0.0 end)) - (Enum.min(succ_scores, fn -> 0.0 end))

    filtered_atoms = if succ_range > 1.0 do
      top_indices =
        succ_scored
        |> Enum.sort_by(fn {_i, r, _} -> -r end)
        |> Enum.take(succ_top_k)
        |> Enum.map(fn {i, _, _} -> i end)
        |> MapSet.new()

      filtered = Enum.with_index(atoms) |> Enum.filter(fn {_, i} -> MapSet.member?(top_indices, i) end) |> Enum.map(fn {a, _} -> a end)
      IO.puts("    Successor pre-filter: #{length(atoms)} → #{length(filtered)} (range=#{Float.round(succ_range, 1)})")
      filtered
    else
      IO.puts("    Successor scores undifferentiated (range=#{Float.round(succ_range, 2)}), using all candidates")
      atoms
    end

    IO.puts("    Scoring #{length(filtered_atoms)} candidates with episodes...")
    {scored, baseline, _} =
      GymOracle.score_candidates(filtered_atoms, action, default, chain_before,
        seeds: seeds, chain_after: chain_after, env: env, max_steps: max_steps)

    IO.puts("    Baseline: #{Float.round(baseline, 1)}")

    best_d0 = Enum.max_by(scored, fn {_i, r, _l} -> r end, fn -> nil end)

    d0_result =
      case best_d0 do
        nil -> nil
        {idx, reward, count} ->
          if reward > baseline do
            pred = Enum.at(filtered_atoms, idx)
            IO.puts("    Best d0: #{format_pred(pred, env)} reward=#{Float.round(reward, 1)} (Δ=#{Float.round(reward - baseline, 1)})")
            {pred, reward, count, scored}
          else
            nil
          end
      end

    if max_depth == 0 do
      case d0_result do
        nil -> nil
        {p, r, l, _} -> {p, r, l}
      end
    else
      top_atoms =
        scored
        |> Enum.sort_by(fn {_idx, r, _l} -> -r end)
        |> Enum.take(top_k)
        |> Enum.map(fn {idx, _r, _l} -> Enum.at(filtered_atoms, idx) end)

      negations = Enum.map(top_atoms, fn p -> {:not, p} end)

      d1_candidates =
        (for p <- top_atoms, q <- top_atoms, p != q, do: {:and, p, q}) ++
        (for p <- top_atoms, q <- top_atoms, p != q, do: {:or, p, q}) ++
        (for p <- negations, q <- top_atoms, do: {:and, p, q}) ++
        (for p <- negations, q <- top_atoms, do: {:or, p, q})

      d1_candidates = Enum.uniq(d1_candidates)
      IO.puts("    Depth 1: #{length(d1_candidates)} candidates")

      {scored_d1, _, _} =
        GymOracle.score_candidates(d1_candidates, action, default, chain_before,
          seeds: seeds, chain_after: chain_after, env: env, max_steps: max_steps)

      best_d1 = Enum.max_by(scored_d1, fn {_i, r, _l} -> r end, fn -> nil end)

      threshold = case d0_result do
        nil -> baseline
        {_, r, _, _} -> r
      end

      case best_d1 do
        nil ->
          case d0_result do
            nil -> nil
            {p, r, l, _} -> {p, r, l}
          end

        {idx, reward, count} ->
          if reward > threshold do
            pred = Enum.at(d1_candidates, idx)
            IO.puts("    Best d1: #{format_pred(pred, env)} reward=#{Float.round(reward, 1)}")
            {pred, reward, count}
          else
            case d0_result do
              nil -> nil
              {p, r, l, _} -> {p, r, l}
            end
          end
      end
    end
  end

  # ── Pretty-printing ────────────────────────────────────────────

  @tetris_dims %{0 => "px", 1 => "py", 2 => "rot", 3 => "vis",
                 4 => "fill", 5 => "bot", 6 => "rows", 7 => "pcs"}

  @tetris_v2_dims %{0 => "r0", 1 => "r1", 2 => "r2", 3 => "r3", 4 => "r4",
                    5 => "r5", 6 => "r6", 7 => "r7", 8 => "r8", 9 => "r9",
                    10 => "r10", 11 => "r11", 12 => "r12", 13 => "r13",
                    14 => "r14", 15 => "r15", 16 => "r16", 17 => "r17",
                    18 => "r18", 19 => "r19", 20 => "px", 21 => "py", 22 => "rot"}

  @ll_dims %{0 => "x", 1 => "y", 2 => "vx", 3 => "vy", 4 => "θ", 5 => "ω"}
  @pend_dims %{0 => "cosθ", 1 => "sinθ", 2 => "ω"}
  @cp_dims %{0 => "x", 1 => "ẋ", 2 => "θ", 3 => "θ̇"}

  defp dim_name(:tetris, d), do: @tetris_dims[d] || "d#{d}"
  defp dim_name(:tetris_v2, d), do: @tetris_v2_dims[d] || "d#{d}"
  defp dim_name(:pendulum, d), do: @pend_dims[d] || "d#{d}"
  defp dim_name(:cartpole, d), do: @cp_dims[d] || "d#{d}"
  defp dim_name(_, d), do: @ll_dims[d] || "d#{d}"

  defp format_pred(:truep, _env), do: "true"
  defp format_pred(:falsep, _env), do: "false"
  defp format_pred({:feat, ["axis", d, t]}, env), do: "#{dim_name(env, d)}<#{t}"
  defp format_pred({:feat, ["diag", i, j, c]}, env), do: "#{c}·#{dim_name(env, i)}+#{dim_name(env, j)}<0"
  defp format_pred({:feat, ["sq_diag", i, j, c]}, env), do: "#{c}·#{dim_name(env, i)}²+#{dim_name(env, j)}<0"
  defp format_pred({:feat, ["prod", i, j, t]}, env), do: "#{dim_name(env, i)}·#{dim_name(env, j)}<#{t}"
  defp format_pred({:not, p}, env), do: "¬(#{format_pred(p, env)})"
  defp format_pred({:and, p, q}, env), do: "(#{format_pred(p, env)} ∧ #{format_pred(q, env)})"
  defp format_pred({:or, p, q}, env), do: "(#{format_pred(p, env)} ∨ #{format_pred(q, env)})"
  defp format_pred(other, _env), do: inspect(other)

  @tetris_py %{0 => "piece_x", 1 => "piece_y", 2 => "rot_raw", 3 => "piece_vis",
               4 => "board_fill", 5 => "bottom_fill", 6 => "occupied_rows", 7 => "pieces_placed"}

  defp dim_py(:tetris, d), do: @tetris_py[d] || "obs[#{d}]"
  defp dim_py(_, d), do: "obs[#{d}]"

  defp print_deployable(chain, default, env) do
    IO.puts("\n=== DEPLOYABLE POLICY ===")
    IO.puts("def policy(obs):")

    Enum.each(chain, fn {pred, action} ->
      IO.puts("    if #{fmt_py(pred, env)}: return #{inspect(action)}")
    end)

    IO.puts("    return #{inspect(default)}")
  end

  defp fmt_py({:feat, ["axis", d, t]}, env), do: "#{dim_py(env, d)} < #{t}"
  defp fmt_py({:feat, ["diag", i, j, c]}, env), do: "#{c}*#{dim_py(env, i)} + #{dim_py(env, j)} < 0"
  defp fmt_py({:feat, ["sq_diag", i, j, c]}, env), do: "#{c}*#{dim_py(env, i)}**2 + #{dim_py(env, j)} < 0"
  defp fmt_py({:feat, ["prod", i, j, t]}, env), do: "#{dim_py(env, i)} * #{dim_py(env, j)} < #{t}"
  defp fmt_py({:not, p}, env), do: "not (#{fmt_py(p, env)})"
  defp fmt_py({:and, p, q}, env), do: "(#{fmt_py(p, env)}) and (#{fmt_py(q, env)})"
  defp fmt_py({:or, p, q}, env), do: "(#{fmt_py(p, env)}) or (#{fmt_py(q, env)})"
  defp fmt_py(other, _env), do: inspect(other)
end
