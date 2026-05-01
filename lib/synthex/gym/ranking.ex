defmodule Synthex.Gym.Ranking do
  @moduledoc """
  CSHRL-native ranking synthesis via flat partition CEGAR.

  The policy is a flat set of (predicate, ranking) pairs plus a default
  ranking.  Each state should satisfy at most one predicate.  Overlapping
  predicates with different rankings are detected as conflicts.  The
  CEGAR loop resolves conflicts by narrowing overly broad predicates.

  Two operations per round:
    - DEFAULT-region CEX  →  add a new (predicate, ranking) pair
    - PREDICATE-region CEX → narrow the predicate via conjunction

  Partition :: [{pred, ranking}, ...]
  """

  alias Synthex.Gym.Oracle, as: GymOracle
  alias Synthex.Core.CEGIS

  def solve(actions, opts \\ []) do
    env = Keyword.fetch!(opts, :env)

    if episodic_oracle?(env) do
      solve_coorddescent(actions, opts)
    else
      solve_cegar(actions, opts)
    end
  end

  # ── Coordinate descent + full ranking (episodic envs) ────────

  defp solve_coorddescent(actions, opts) do
    env = Keyword.fetch!(opts, :env)
    max_steps = Keyword.fetch!(opts, :max_steps)

    action_names = actions |> Enum.with_index() |> Map.new(fn {a, i} -> {i, a} end)
    n_actions = length(actions)
    val_seeds = Enum.to_list(10_000..10_499)
    ranking_seeds = Enum.to_list(20_000..20_499)

    IO.puts("══════════════════════════════════════════════════════")
    IO.puts("  CSHRL Ranking Synthesis — Coord Descent + Full Ranking")
    IO.puts("  Env: #{env}")
    IO.puts("  Actions: #{inspect(actions)} (#{n_actions}! = #{factorial(n_actions)} permutations)")
    IO.puts("══════════════════════════════════════════════════════\n")

    # Phase 1: Chain coordinate descent to find predicates
    IO.puts("═══ Phase 1: Coordinate Descent (predicate discovery) ═══\n")

    default_action = List.last(actions)
    chain_priority = Enum.drop(actions, -1)

    chain_opts = [
      env: env,
      max_steps: max_steps,
      depth: Keyword.get(opts, :depth, 1),
      max_coeff: Keyword.get(opts, :max_coeff, 5),
      n_episodes: Keyword.get(opts, :n_episodes, 200),
      top_k: Keyword.get(opts, :top_k, 30),
      max_iters: Keyword.get(opts, :max_iters, 5),
      cegar_rounds: Keyword.get(opts, :cegar_rounds, 3)
    ]

    {chain, _default} = Synthex.Gym.Chain.solve(chain_priority, default_action, chain_opts)

    active_chain = Enum.reject(chain, fn {p, _} -> p == :falsep end)

    if active_chain == [] do
      IO.puts("\n  No predicates found — nothing to rank.")
      default_ranking = Enum.to_list(0..(n_actions - 1))
      print_partition_summary([], default_ranking, action_names, val_seeds, env, max_steps)
    else
      # Phase 2: Full ranking per region via episode reward
      IO.puts("\n═══ Phase 2: Full Ranking Determination ═══\n")
      IO.puts("  Scoring all #{n_actions} actions per region (#{length(ranking_seeds)} episodes each)...\n")

      region_rankings =
        active_chain
        |> Enum.with_index()
        |> Enum.map(fn {{pred, _top_action}, pos} ->
          IO.puts("  Region #{pos}: #{format_pred(pred, env)}")

          action_rewards =
            Enum.map(0..(n_actions - 1), fn aidx ->
              action_name = Map.get(action_names, aidx)
              modified_chain = List.replace_at(active_chain, pos, {pred, action_name})
              serialized = GymOracle.serialize_chain(modified_chain, env)
              default_int = GymOracle.serialize_action(default_action, env)

              result = call_python(%{
                "cmd" => "score",
                "candidates" => [],
                "stage_action" => 0,
                "default" => default_int,
                "chain_so_far" => serialized,
                "chain_after" => [],
                "seeds" => ranking_seeds,
                "max_steps" => max_steps
              }, env)

              reward = result["baseline_reward"]
              landings = result["baseline_landings"] || 0
              avg = Float.round(reward / length(ranking_seeds), 1)
              IO.puts("    #{inspect(action_name)}: avg=#{avg}, landings=#{landings}/#{length(ranking_seeds)}")
              {aidx, reward}
            end)

          ranking =
            action_rewards
            |> Enum.sort_by(fn {_idx, reward} -> -reward end)
            |> Enum.map(fn {idx, _} -> idx end)

          named = Enum.map(ranking, &Map.get(action_names, &1))
          IO.puts("    → #{format_ranking(named)}\n")
          {pred, ranking}
        end)

      # Default region ranking
      IO.puts("  Default region:")

      default_rewards =
        Enum.map(0..(n_actions - 1), fn aidx ->
          action_name = Map.get(action_names, aidx)
          serialized = GymOracle.serialize_chain(active_chain, env)
          default_int = GymOracle.serialize_action(action_name, env)

          result = call_python(%{
            "cmd" => "score",
            "candidates" => [],
            "stage_action" => 0,
            "default" => default_int,
            "chain_so_far" => serialized,
            "chain_after" => [],
            "seeds" => ranking_seeds,
            "max_steps" => max_steps
          }, env)

          reward = result["baseline_reward"]
          landings = result["baseline_landings"] || 0
          avg = Float.round(reward / length(ranking_seeds), 1)
          IO.puts("    #{inspect(action_name)}: avg=#{avg}, landings=#{landings}/#{length(ranking_seeds)}")
          {aidx, reward}
        end)

      default_ranking =
        default_rewards
        |> Enum.sort_by(fn {_idx, reward} -> -reward end)
        |> Enum.map(fn {idx, _} -> idx end)

      named_default = Enum.map(default_ranking, &Map.get(action_names, &1))
      IO.puts("    → #{format_ranking(named_default)}\n")

      print_partition_summary(region_rankings, default_ranking, action_names, val_seeds, env, max_steps)
    end
  end

  # ── Flat-partition CEGAR (non-episodic envs) ─────────────────

  defp solve_cegar(actions, opts) do
    env = Keyword.fetch!(opts, :env)
    max_steps = Keyword.fetch!(opts, :max_steps)
    depth = Keyword.get(opts, :depth, 1)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    lookahead = Keyword.get(opts, :lookahead, 20)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 15)
    top_k = Keyword.get(opts, :top_k, 30)
    n_verify = Keyword.get(opts, :n_verify, 200)

    action_names = actions |> Enum.with_index() |> Map.new(fn {a, i} -> {i, a} end)
    n_actions = length(actions)
    val_seeds = Enum.to_list(10_000..10_499)
    episodic? = false

    IO.puts("══════════════════════════════════════════════════════")
    IO.puts("  CSHRL Ranking Synthesis — Flat Partition CEGAR")
    IO.puts("  Env: #{env}")
    IO.puts("  Actions: #{inspect(actions)} (#{n_actions}! = #{factorial(n_actions)} permutations)")
    IO.puts("  Depth: #{depth}, Lookahead: #{lookahead}")
    IO.puts("  CEGAR rounds: #{cegar_rounds}, Oracle verify top: #{n_verify}")
    IO.puts("  max_steps: #{max_steps}")
    IO.puts("══════════════════════════════════════════════════════\n")

    # 1. Initial anchors + default ranking
    {anchors, default_ranking} =
      if episodic? do
        IO.puts("  Bootstrap: discovering initial default ranking...")
        {states, rankings} = collect_and_profile_random(
          Enum.to_list(0..49), min(max_steps, 200), 1, env)
        IO.puts("  #{length(states)} bootstrap samples (lookahead=1, heuristic only)")
        dr = majority(rankings)
        named = Enum.map(dr, &Map.get(action_names, &1))
        IO.puts("  Default ranking: #{format_ranking(named)}")
        {states, dr}
      else
        IO.puts("  Collecting initial anchors...")
        anch = collect_anchors(Enum.to_list(0..19), 100, 5, env)
        IO.puts("  #{length(anch)} initial anchors")
        init_lookahead = min(lookahead, 20)
        IO.puts("  Profiling for initial default ranking (self-consistent, lookahead=#{init_lookahead})...")
        {dr, named} =
          iterate_initial_ranking(anch, actions, action_names, init_lookahead, env, hd(actions), 0)
        IO.puts("  Default ranking: #{format_ranking(named)}")
        {anch, dr}
      end

    partition = []

    # 2. Iterative CEGAR
    {_final_part, _final_default, _final_anchors, best_part, best_default, best_successes} =
      Enum.reduce_while(1..cegar_rounds,
        {partition, default_ranking, anchors, partition, default_ranking, 0},
        fn round, {part, default, anchors, best_part, best_default, best_succ} ->

        n_regions = length(part) + 1
        IO.puts("\n████████ CEGAR Round #{round}/#{cegar_rounds} — #{n_regions} regions ████████")

        effective_lookahead = min(lookahead, max(20, n_regions * 12))
        if effective_lookahead != lookahead do
          IO.puts("    [Progressive: lookahead=#{effective_lookahead} (partition too small for #{lookahead})]")
        end

        IO.puts("  Oracle verifying #{n_regions}-region partition...")
        t0 = System.monotonic_time(:millisecond)

        {current_anchors, oracle_rankings} =
          if episodic? do
            verify_seeds = Enum.to_list((round * 20)..((round + 1) * 20 - 1))
            IO.puts("    [Partition-policy follow-up, lookahead=0 (full episode)]")
            oracle_verify_partition(part, default, verify_seeds, max_steps, 0, env)
          else
            rankings = oracle_verify_preds(part, default, anchors, effective_lookahead, env)
            {anchors, rankings}
          end

        t1 = System.monotonic_time(:millisecond)
        IO.puts("  Oracle done (#{t1 - t0}ms) — #{length(current_anchors)} profile points")

        {part_after_overlaps, n_overlaps_fixed} =
          resolve_overlaps(part, current_anchors, oracle_rankings, action_names, env,
                           depth, max_coeff, top_k, n_verify)

        if n_overlaps_fixed > 0 do
          IO.puts("  Fixed #{n_overlaps_fixed} overlap conflicts")
        end

        region_data = classify_by_region(part_after_overlaps, default, current_anchors, oracle_rankings)

        total = length(current_anchors)
        total_ok = Enum.reduce(region_data, 0, fn {_region, data}, acc -> acc + length(data.ok) end)
        total_cex = Enum.reduce(region_data, 0, fn {_region, data}, acc -> acc + length(data.cex) end)
        pct = Float.round(total_ok / max(total, 1) * 100, 1)

        remaining_conflicts = count_conflicts(part_after_overlaps, current_anchors)
        IO.puts("  Consistent: #{total_ok}/#{total} (#{pct}%)")
        if remaining_conflicts > 0 do
          IO.puts("  ⚠ #{remaining_conflicts} residual overlap conflicts")
        end

        if total_cex == 0 and remaining_conflicts == 0 do
          IO.puts("  ★ Fully self-consistent! CEGAR converged.")
          {:halt, {part_after_overlaps, default, anchors, best_part, best_default, best_succ}}
        else
          IO.puts("  #{total_cex} CEX total:")
          Enum.each(region_data, fn {region, data} ->
            if data.cex != [] do
              region_name = if region == :default, do: "default", else: "region #{region}"
              cex_by_ranking =
                data.cex
                |> Enum.group_by(fn {_s, oracle_r} -> oracle_r end)
                |> Enum.sort_by(fn {_r, items} -> -length(items) end)
              IO.puts("    #{region_name} (#{length(data.ok)} ok, #{length(data.cex)} cex):")
              Enum.each(Enum.take(cex_by_ranking, 3), fn {ranking, items} ->
                named = Enum.map(ranking, &Map.get(action_names, &1))
                IO.puts("      needs #{format_ranking(named)}: #{length(items)}")
              end)
            end
          end)

          regions_by_severity =
            region_data
            |> Enum.filter(fn {_, data} -> data.cex != [] end)
            |> Enum.sort_by(fn {_, data} -> -length(data.cex) end)

          {new_part, action_taken} =
            if n_overlaps_fixed > 0 do
              {part_after_overlaps, true}
            else
              Enum.reduce_while(regions_by_severity, {part_after_overlaps, false}, fn {region, data}, {cur_part, _} ->
                {result_part, success} =
                  if region == :default do
                    handle_default_cex(cur_part, default, data, current_anchors, oracle_rankings,
                                       action_names, env, depth, max_coeff, top_k, n_verify, max_steps)
                  else
                    handle_predicate_cex(cur_part, region, data,
                                         action_names, env, depth, max_coeff, top_k, n_verify)
                  end

                if success do
                  {:halt, {result_part, true}}
                else
                  {:cont, {cur_part, false}}
                end
              end)
            end

          effective_part = if action_taken, do: new_part, else: part

          {new_best_part, new_best_default, new_best_succ} =
            if action_taken do
              {val_reward, val_count} = validate_partition(new_part, default, action_names, val_seeds, env, max_steps)
              IO.puts("  ▸ Validation: reward=#{Float.round(val_reward / max(length(val_seeds), 1), 1)} avg, successes=#{val_count}/#{length(val_seeds)}")
              IO.puts("  Partition now has #{length(new_part) + 1} regions")

              if val_count >= best_succ do
                if val_count > best_succ do
                  IO.puts("  ★★ New best! #{val_count}/#{length(val_seeds)} (was #{best_succ})")
                end
                {new_part, default, val_count}
              else
                {best_part, best_default, best_succ}
              end
            else
              IO.puts("  No progress this round")
              {best_part, best_default, best_succ}
            end

          new_anchors =
            if episodic? do
              current_anchors
            else
              cex_seeds = Enum.to_list((round * 20)..((round + 1) * 20 - 1))
              cex = find_cex_partition(effective_part, default, cex_seeds, max_steps, lookahead, env)
              IO.puts("  +#{length(cex)} fresh counterexample states")
              deduplicate_anchors(anchors ++ cex)
            end

          {:cont, {effective_part, default, new_anchors, new_best_part, new_best_default, new_best_succ}}
        end
      end)

    IO.puts("\n  Best partition: #{best_successes}/#{length(val_seeds)} successes (#{length(best_part) + 1} regions)")
    print_partition_summary(best_part, best_default, action_names, val_seeds, env, max_steps)
  end

  defp episodic_oracle?(:lunarlander), do: true
  defp episodic_oracle?(:pong), do: true
  defp episodic_oracle?(:cartpole), do: true
  defp episodic_oracle?(:acrobot), do: true
  defp episodic_oracle?(:mountaincar), do: true
  defp episodic_oracle?(:breakout), do: true
  defp episodic_oracle?(_), do: false

  # ── Region classification ─────────────────────────────────────

  defp classify_by_region(partition, default, anchors, oracle_rankings) do
    initial = %{default: %{ok: [], cex: []}}

    initial =
      if partition == [] do
        initial
      else
        Enum.reduce(0..(length(partition) - 1), initial, fn i, acc ->
          Map.put(acc, i, %{ok: [], cex: []})
        end)
      end

    Enum.zip(anchors, oracle_rankings)
    |> Enum.reduce(initial, fn {state, oracle_r}, acc ->
      region = find_region(partition, state)
      assigned_ranking =
        case region do
          :default -> default
          i -> elem(Enum.at(partition, i), 1)
        end

      data = Map.get(acc, region)
      if assigned_ranking == oracle_r do
        Map.put(acc, region, %{data | ok: [{state, oracle_r} | data.ok]})
      else
        Map.put(acc, region, %{data | cex: [{state, oracle_r} | data.cex]})
      end
    end)
  end

  defp find_region(partition, state) do
    result =
      partition
      |> Enum.with_index()
      |> Enum.find(fn {{pred, _ranking}, _idx} -> GymOracle.eval_pred(pred, state) end)

    case result do
      nil -> :default
      {_, idx} -> idx
    end
  end

  defp count_conflicts(partition, anchors) do
    Enum.count(anchors, fn state ->
      matching =
        partition
        |> Enum.filter(fn {pred, _} -> GymOracle.eval_pred(pred, state) end)
        |> Enum.map(fn {_, r} -> r end)
        |> Enum.uniq()
      length(matching) > 1
    end)
  end

  defp resolve_overlaps(partition, anchors, oracle_rankings, action_names, env,
                        depth, max_coeff, top_k, n_verify) do
    conflicts =
      Enum.zip(anchors, oracle_rankings)
      |> Enum.flat_map(fn {state, oracle_r} ->
        matching =
          partition
          |> Enum.with_index()
          |> Enum.filter(fn {{pred, _}, _idx} -> GymOracle.eval_pred(pred, state) end)

        rankings = matching |> Enum.map(fn {{_, r}, _} -> r end) |> Enum.uniq()

        if length(rankings) > 1 do
          wrong_indices =
            matching
            |> Enum.filter(fn {{_, r}, _} -> r != oracle_r end)
            |> Enum.map(fn {_, idx} -> idx end)
          [{state, oracle_r, wrong_indices}]
        else
          []
        end
      end)

    if conflicts == [] do
      {partition, 0}
    else
      wrong_by_pred =
        conflicts
        |> Enum.flat_map(fn {state, _oracle_r, wrong_indices} ->
          Enum.map(wrong_indices, fn idx -> {idx, state} end)
        end)
        |> Enum.group_by(fn {idx, _} -> idx end, fn {_, state} -> state end)

      IO.puts("\n  Resolving #{length(conflicts)} overlap conflicts:")
      Enum.each(wrong_by_pred, fn {idx, states} ->
        {_pred, ranking} = Enum.at(partition, idx)
        named = Enum.map(ranking, &Map.get(action_names, &1))
        IO.puts("    region #{idx} (#{format_ranking(named)}): wrong at #{length(states)} overlapping states")
      end)

      {worst_idx, worst_states} =
        wrong_by_pred
        |> Enum.max_by(fn {_idx, states} -> length(states) end)

      {old_pred, old_ranking} = Enum.at(partition, worst_idx)
      old_named = Enum.map(old_ranking, &Map.get(action_names, &1))

      ok_in_region =
        Enum.zip(anchors, oracle_rankings)
        |> Enum.filter(fn {state, oracle_r} ->
          GymOracle.eval_pred(old_pred, state) and oracle_r == old_ranking
        end)
        |> Enum.map(fn {s, _} -> s end)

      IO.puts("  → Narrowing region #{worst_idx} (#{format_ranking(old_named)}) to exclude #{length(worst_states)} overlap-wrong states")
      IO.puts("    keeping #{length(ok_in_region)} correct states")

      cond do
        length(ok_in_region) == 0 ->
          IO.puts("    ✗ No correct states remain — removing region")
          {List.delete_at(partition, worst_idx), length(conflicts)}

        length(ok_in_region) < 3 ->
          IO.puts("    ✗ Too few correct states (#{length(ok_in_region)}) — removing region")
          {List.delete_at(partition, worst_idx), length(conflicts)}

        true ->
          all_states = ok_in_region ++ worst_states
          best = search_predicate(ok_in_region, worst_states, all_states, env,
                                  depth, max_coeff, top_k, n_verify)

          case best do
            nil ->
              IO.puts("    No narrowing predicate found — removing region")
              {List.delete_at(partition, worst_idx), length(conflicts)}
            {filter_pred, score} ->
              narrowed = {:and, old_pred, filter_pred}
              IO.puts("    ★ Narrowed: #{format_pred(narrowed, env)}")
              IO.puts("      keeps #{Float.round(score * 100, 1)}% of correct, excludes overlap-wrong")
              {List.replace_at(partition, worst_idx, {narrowed, old_ranking}), length(conflicts)}
          end
      end
    end
  end

  # ── Handle default-region CEX: add new predicate ──────────────

  defp handle_default_cex(partition, default, worst_data, all_anchors, _all_oracle_rankings,
                          action_names, env, depth, max_coeff, top_k, n_verify, max_steps) do
    cex_by_ranking =
      worst_data.cex
      |> Enum.group_by(fn {_s, oracle_r} -> oracle_r end)
      |> Enum.sort_by(fn {_r, items} -> -length(items) end)

    ok_states = Enum.map(worst_data.ok, fn {s, _} -> s end)

    all_observed_rankings =
      worst_data.cex
      |> Enum.map(fn {_s, r} -> r end)
      |> Enum.uniq()

    IO.puts("  #{length(all_observed_rankings)} distinct rankings in CEX")

    all_preds =
      Enum.flat_map(cex_by_ranking, fn {target_ranking, target_items} ->
        target_cex = Enum.map(target_items, fn {s, _} -> s end)
        target_named = Enum.map(target_ranking, &Map.get(action_names, &1))

        other_cex = worst_data.cex
          |> Enum.filter(fn {_, r} -> r != target_ranking end)
          |> Enum.map(fn {s, _} -> s end)
        exclude = ok_states ++ other_cex

        IO.puts("  → Pre-filtering for #{length(target_cex)} default-CEX (#{format_ranking(target_named)})")

        top_n = search_predicate_topn(target_cex, exclude, all_anchors, env, depth, max_coeff, top_k, n_verify)
                |> Enum.take(10)

        IO.puts("    #{length(top_n)} pre-filtered candidates")
        Enum.map(top_n, fn {pred, f1} -> {pred, f1} end)
      end)
      |> Enum.uniq_by(fn {pred, _f1} -> pred end)

    if all_preds == [] do
      IO.puts("    No candidates from any ranking group")
      {partition, false}
    else
      all_reward_candidates =
        for {pred, f1} <- all_preds,
            ranking <- all_observed_rankings,
            do: {pred, ranking, f1}

      IO.puts("  Scoring #{length(all_preds)} predicates × #{length(all_observed_rankings)} rankings = #{length(all_reward_candidates)} combos by episode reward...")

      {best_pred, best_ranking, best_reward, best_count} =
        score_partition_candidates(partition, default, all_reward_candidates, action_names, env, max_steps)

      if best_pred == nil do
        IO.puts("    No candidate improved over baseline")
        {partition, false}
      else
        best_named = Enum.map(best_ranking, &Map.get(action_names, &1))
        IO.puts("  ★ New region: #{format_pred(best_pred, env)}")
        IO.puts("    → #{format_ranking(best_named)} (reward=#{Float.round(best_reward, 1)}, successes=#{best_count})")

        {partition ++ [{best_pred, best_ranking}], true}
      end
    end
  end

  # ── Handle predicate-region CEX: narrow existing predicate ────

  defp handle_predicate_cex(partition, region_idx, worst_data,
                            action_names, env, depth, max_coeff, top_k, n_verify) do
    {old_pred, old_ranking} = Enum.at(partition, region_idx)
    old_named = Enum.map(old_ranking, &Map.get(action_names, &1))

    ok_states = Enum.map(worst_data.ok, fn {s, _} -> s end)
    cex_states = Enum.map(worst_data.cex, fn {s, _} -> s end)

    IO.puts("  → Narrowing region #{region_idx} (#{format_ranking(old_named)})")
    IO.puts("    #{length(ok_states)} ok + #{length(cex_states)} cex in region")

    cond do
      length(ok_states) == 0 ->
        IO.puts("  ✗ Region has 0 correct states — removing it")
        new_part = List.delete_at(partition, region_idx)
        {new_part, true}

      length(ok_states) < 3 ->
        IO.puts("  ✗ Region has only #{length(ok_states)} correct states — removing it")
        new_part = List.delete_at(partition, region_idx)
        {new_part, true}

      true ->
        all_states = ok_states ++ cex_states
        best = search_predicate(ok_states, cex_states, all_states, env, depth, max_coeff, top_k, n_verify)

        case best do
          nil ->
            IO.puts("  No narrowing predicate found")
            {partition, false}
          {filter_pred, score} ->
            narrowed = {:and, old_pred, filter_pred}
            IO.puts("  ★ Narrowed: #{format_pred(narrowed, env)}")
            IO.puts("    keeps #{Float.round(score * 100, 1)}% of OK, excludes CEX")

            new_part = List.replace_at(partition, region_idx, {narrowed, old_ranking})
            {new_part, true}
        end
    end
  end

  # ── Predicate search ──────────────────────────────────────────

  defp search_predicate(target_states, exclude_states, all_states, env, depth, max_coeff, top_k, n_verify) do
    case search_predicate_topn(target_states, exclude_states, all_states, env, depth, max_coeff, top_k, n_verify) do
      [] -> nil
      [{pred, score} | _] -> {pred, score}
    end
  end

  defp search_predicate_topn(target_states, exclude_states, all_states, env, depth, max_coeff, top_k, n_verify) do
    feats = GymOracle.generate_features(all_states, env: env, max_coeff: max_coeff)
    atoms = CEGIS.enumerate(feats, 0)

    d0_filtered = pre_filter_split(atoms, target_states, exclude_states)
    IO.puts("  #{length(d0_filtered)} discriminating d0 atoms")

    d1_filtered =
      if depth >= 1 and d0_filtered != [] do
        top_atoms = d0_filtered |> Enum.take(top_k) |> Enum.map(fn {pred, _} -> pred end)
        negations = Enum.map(top_atoms, fn p -> {:not, p} end)
        d1_preds =
          (for p <- top_atoms, q <- top_atoms, p != q, do: {:and, p, q}) ++
          (for p <- top_atoms, q <- top_atoms, p != q, do: {:or, p, q}) ++
          (for p <- negations, q <- top_atoms, do: {:and, p, q}) ++
          (for p <- negations, q <- top_atoms, do: {:or, p, q})
          |> Enum.uniq()
        d1_result = pre_filter_split(d1_preds, target_states, exclude_states)
        IO.puts("  #{length(d1_result)} discriminating d1 predicates")
        d1_result
      else
        []
      end

    d2_filtered =
      if depth >= 2 and d1_filtered != [] do
        top_d1 = d1_filtered |> Enum.take(top_k) |> Enum.map(fn {pred, _} -> pred end)
        top_d0 = d0_filtered |> Enum.take(top_k) |> Enum.map(fn {pred, _} -> pred end)
        d0_neg = Enum.map(top_d0, fn p -> {:not, p} end)
        d2_preds =
          (for p <- top_d1, q <- top_d0, do: {:and, p, q}) ++
          (for p <- top_d1, q <- d0_neg, do: {:and, p, q})
          |> Enum.uniq()
        d2_result = pre_filter_split(d2_preds, target_states, exclude_states)
        IO.puts("  #{length(d2_result)} discriminating d2 predicates")
        d2_result
      else
        []
      end

    all_candidates =
      (d0_filtered ++ d1_filtered ++ d2_filtered)
      |> Enum.sort_by(fn {_, score} -> -score end)
      |> Enum.uniq_by(fn {pred, _} -> pred end)
      |> Enum.take(n_verify)

    all_candidates
    |> Enum.map(fn {pred, _pre_score} ->
      matched_target = Enum.count(target_states, &GymOracle.eval_pred(pred, &1))
      matched_exclude = Enum.count(exclude_states, &GymOracle.eval_pred(pred, &1))
      total_matched = matched_target + matched_exclude

      precision = if total_matched > 0, do: matched_target / total_matched, else: 0
      recall = if target_states != [], do: matched_target / length(target_states), else: 0

      if precision < 0.5 or recall < 0.1 or matched_target < 2 do
        nil
      else
        f1 = 2 * precision * recall / (precision + recall)
        {pred, f1}
      end
    end)
    |> Enum.reject(&is_nil/1)
    |> Enum.sort_by(fn {_, score} -> -score end)
  end

  defp check_new_overlaps(new_pred, new_ranking, partition, states) do
    Enum.count(states, fn state ->
      if GymOracle.eval_pred(new_pred, state) do
        Enum.any?(partition, fn {pred, ranking} ->
          ranking != new_ranking and GymOracle.eval_pred(pred, state)
        end)
      else
        false
      end
    end)
  end

  # ── Pre-filter ──────────────────────────────────────────────

  defp pre_filter_split(_candidates, target_states, _exclude_states)
       when target_states == [], do: []

  defp pre_filter_split(candidates, target_states, exclude_states) do
    n_tgt = length(target_states)

    candidates
    |> Enum.reject(fn pred -> pred == :truep or pred == :falsep end)
    |> Enum.map(fn pred ->
      tgt_true = Enum.count(target_states, &GymOracle.eval_pred(pred, &1))
      tgt_false = n_tgt - tgt_true
      exc_true = Enum.count(exclude_states, &GymOracle.eval_pred(pred, &1))
      exc_false = length(exclude_states) - exc_true

      precision_a = if tgt_true + exc_true > 0, do: tgt_true / (tgt_true + exc_true), else: 0
      recall_a = tgt_true / n_tgt
      precision_b = if tgt_false + exc_false > 0, do: tgt_false / (tgt_false + exc_false), else: 0
      recall_b = tgt_false / n_tgt

      score_a = if precision_a > 0 and recall_a > 0, do: 2 * precision_a * recall_a / (precision_a + recall_a), else: 0
      score_b = if precision_b > 0 and recall_b > 0, do: 2 * precision_b * recall_b / (precision_b + recall_b), else: 0

      best_score = max(score_a, score_b)
      final_pred = if score_b > score_a, do: {:not, pred}, else: pred

      if best_score > 0.1, do: {final_pred, best_score}, else: nil
    end)
    |> Enum.reject(&is_nil/1)
    |> Enum.sort_by(fn {_, score} -> -score end)
  end

  # ── Oracle interface (state restoration) ───────────────────

  defp oracle_verify_preds(partition, default, anchors, lookahead, env) do
    serialized_preds = Enum.map(partition, fn {pred, ranking} ->
      %{"pred" => GymOracle.serialize_pred(pred), "ranking" => ranking}
    end)

    candidate = %{"preds" => serialized_preds, "default_ranking" => default}

    result = call_python(%{
      "cmd" => "oracle_verify_multi",
      "candidates" => [candidate],
      "anchors" => anchors,
      "lookahead" => lookahead
    }, env)

    [%{"rankings" => rankings}] = result["results"]
    rankings
  end

  # ── Oracle interface (episodic — replay-based) ─────────────

  defp collect_and_profile_random(seeds, max_steps, lookahead, env) do
    result = call_python(%{
      "cmd" => "collect_and_profile",
      "seeds" => seeds,
      "max_steps" => max_steps,
      "lookahead" => lookahead,
      "sample_interval" => 10
    }, env)

    {result["states"], result["rankings"]}
  end

  defp oracle_verify_partition(partition, default, seeds, max_steps, lookahead, env) do
    serialized_preds = Enum.map(partition, fn {pred, ranking} ->
      %{"pred" => GymOracle.serialize_pred(pred), "ranking" => ranking}
    end)

    result = call_python(%{
      "cmd" => "oracle_verify_episodic",
      "preds" => serialized_preds,
      "default_ranking" => default,
      "seeds" => seeds,
      "max_steps" => max_steps,
      "lookahead" => lookahead,
      "sample_interval" => 10
    }, env)

    {result["states"], result["rankings"]}
  end

  # ── Other oracle helpers ───────────────────────────────────

  defp validate_partition(partition, default, action_names, val_seeds, env, max_steps) do
    chain = Enum.map(partition, fn {pred, ranking} ->
      named_top = Map.get(action_names, hd(ranking))
      {pred, named_top}
    end)
    serialized = GymOracle.serialize_chain(chain, env)
    default_named_top = Map.get(action_names, hd(default))
    default_int = GymOracle.serialize_action(default_named_top, env)

    request = %{
      "cmd" => "score",
      "candidates" => [],
      "stage_action" => 0,
      "default" => default_int,
      "chain_so_far" => serialized,
      "chain_after" => [],
      "seeds" => val_seeds,
      "max_steps" => max_steps
    }

    result = call_python(request, env)
    {result["baseline_reward"], result["baseline_landings"] || result["n_stabilized"] || 0}
  end

  defp score_partition_candidates(partition, default, candidates, action_names, env, max_steps) do
    n_score_episodes = 100
    seed_offset = :rand.uniform(10_000)
    seeds = Enum.to_list(seed_offset..(seed_offset + n_score_episodes - 1))

    chain_so_far = Enum.map(partition, fn {pred, ranking} ->
      named_top = Map.get(action_names, hd(ranking))
      {pred, named_top}
    end)
    serialized_chain = GymOracle.serialize_chain(chain_so_far, env)

    default_named_top = Map.get(action_names, hd(default))
    default_int = GymOracle.serialize_action(default_named_top, env)

    serialized_candidates =
      Enum.map(candidates, fn {pred, ranking, _f1} ->
        named_top = Map.get(action_names, hd(ranking))
        stage_action = GymOracle.serialize_action(named_top, env)
        {GymOracle.serialize_pred(pred), stage_action}
      end)

    by_action = Enum.group_by(
      Enum.with_index(serialized_candidates),
      fn {{_pred, action}, _idx} -> action end,
      fn {{pred, _action}, idx} -> {pred, idx} end
    )

    scored_results =
      Enum.flat_map(by_action, fn {stage_action, preds_with_idx} ->
        cand_preds = Enum.map(preds_with_idx, fn {pred, _idx} -> pred end)
        orig_indices = Enum.map(preds_with_idx, fn {_pred, idx} -> idx end)

        request = %{
          "cmd" => "score",
          "candidates" => cand_preds,
          "stage_action" => stage_action,
          "default" => default_int,
          "chain_so_far" => serialized_chain,
          "chain_after" => [],
          "seeds" => seeds,
          "max_steps" => max_steps
        }

        result = call_python(request, env)
        baseline = result["baseline_reward"]

        Enum.map(result["scores"], fn s ->
          orig_idx = Enum.at(orig_indices, s["idx"])
          reward = s["reward"]
          count = s["stabilized"] || s["landings"] || 0
          {orig_idx, reward, count, baseline}
        end)
      end)

    case scored_results do
      [] ->
        {nil, nil, nil, 0}
      results ->
        best = Enum.max_by(results, fn {_idx, reward, _count, _baseline} -> reward end)
        {best_idx, best_reward, best_count, baseline} = best

        if best_reward <= baseline do
          IO.puts("    Best candidate reward #{Float.round(best_reward, 1)} <= baseline #{Float.round(baseline, 1)}")
          {nil, nil, nil, 0}
        else
          {pred, ranking, _f1} = Enum.at(candidates, best_idx)
          IO.puts("    Best: reward=#{Float.round(best_reward, 1)} (baseline=#{Float.round(baseline, 1)}, +#{Float.round(best_reward - baseline, 1)})")
          {pred, ranking, best_reward, best_count}
        end
    end
  end

  defp find_cex_partition(partition, default, seeds, max_steps, lookahead, env) do
    serialized_preds = Enum.map(partition, fn {pred, ranking} ->
      %{"pred" => GymOracle.serialize_pred(pred), "ranking" => ranking}
    end)

    request = %{
      "cmd" => "find_cex_multi",
      "preds" => serialized_preds,
      "default_ranking" => default,
      "seeds" => seeds,
      "max_steps" => max_steps,
      "lookahead" => lookahead,
      "max_cex" => 50
    }

    result = call_python(request, env)
    result["cex_states"]
  end

  # ── Init helpers ───────────────────────────────────────────

  defp iterate_initial_ranking(anchors, actions, action_names, lookahead, env, follow_action, iter)
       when iter < 5 do
    rankings = profile_anchors(anchors, [], follow_action, lookahead, env)
    ranking = majority(rankings)
    top_action = Enum.at(actions, hd(ranking))
    named = Enum.map(ranking, &Map.get(action_names, &1))
    n_ok = Enum.count(rankings, &(&1 == ranking))
    pct = Float.round(n_ok / max(length(rankings), 1) * 100, 1)
    IO.puts("    iter #{iter}: #{format_ranking(named)} (#{pct}% under #{inspect(follow_action)})")

    if top_action == follow_action do
      {ranking, named}
    else
      iterate_initial_ranking(anchors, actions, action_names, lookahead, env, top_action, iter + 1)
    end
  end

  defp iterate_initial_ranking(anchors, _actions, action_names, lookahead, env, follow_action, _iter) do
    IO.puts("    (max iterations — using last ranking)")
    rankings = profile_anchors(anchors, [], follow_action, lookahead, env)
    ranking = majority(rankings)
    named = Enum.map(ranking, &Map.get(action_names, &1))
    {ranking, named}
  end

  defp majority(rankings) do
    rankings
    |> Enum.frequencies()
    |> Enum.max_by(fn {_ranking, count} -> count end)
    |> elem(0)
  end

  defp factorial(0), do: 1
  defp factorial(n) when n > 0, do: n * factorial(n - 1)

  # ── Python bridge ──────────────────────────────────────────────

  defp collect_anchors(seeds, max_steps, n_per_episode, env) do
    result = call_python(%{
      "cmd" => "collect_anchors",
      "seeds" => seeds,
      "max_steps" => max_steps,
      "n_per_episode" => n_per_episode
    }, env)
    result["anchors"]
  end

  defp profile_anchors(anchors, chain, default_action, lookahead, env) do
    serialized_chain = GymOracle.serialize_chain(chain, env)
    default_int = GymOracle.serialize_action(default_action, env)

    result = call_python(%{
      "cmd" => "profile_anchors",
      "anchors" => anchors,
      "chain" => serialized_chain,
      "default" => default_int,
      "lookahead" => lookahead
    }, env)
    result["rankings"]
  end

  defp call_python(request, env) do
    script = GymOracle.oracle_script(env)
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_rank_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_rank_resp_#{uid}.json")

    File.write!(req_file, Jason.encode!(request))

    {_output, _exit_code} =
      System.cmd("python3", ["-u", script, req_file, resp_file],
        stderr_to_stdout: true,
        cd: Path.expand("../../..", __DIR__)
      )

    result = Jason.decode!(File.read!(resp_file))
    File.rm(req_file)
    File.rm(resp_file)
    result
  end

  defp deduplicate_anchors(anchors) do
    anchors
    |> Enum.uniq_by(fn state ->
      Enum.map(state, &Float.round(&1 * 1.0, 4))
    end)
  end

  # ── Pretty-printing ────────────────────────────────────────────

  defp format_ranking(ranking), do: ranking |> Enum.map(&inspect/1) |> Enum.join(" > ")

  @pend_dims %{0 => "cosθ", 1 => "sinθ", 2 => "ω"}
  @ll_dims %{0 => "x", 1 => "y", 2 => "vx", 3 => "vy", 4 => "θ", 5 => "ω"}
  @pong_dims %{0 => "bx", 1 => "by", 2 => "py", 3 => "vx", 4 => "vy", 5 => "ey"}
  @cp_dims %{0 => "x", 1 => "ẋ", 2 => "θ", 3 => "θ̇"}
  @acro_dims %{0 => "c1", 1 => "s1", 2 => "c2", 3 => "s2", 4 => "ω1", 5 => "ω2"}
  @mc_dims %{0 => "pos", 1 => "vel"}
  @bo_dims %{0 => "bx", 1 => "by", 2 => "px", 3 => "dx", 4 => "dy"}

  defp dim_name(:pendulum, d), do: @pend_dims[d] || "d#{d}"
  defp dim_name(:pong, d), do: @pong_dims[d] || "d#{d}"
  defp dim_name(:cartpole, d), do: @cp_dims[d] || "d#{d}"
  defp dim_name(:acrobot, d), do: @acro_dims[d] || "d#{d}"
  defp dim_name(:mountaincar, d), do: @mc_dims[d] || "d#{d}"
  defp dim_name(:breakout, d), do: @bo_dims[d] || "d#{d}"
  defp dim_name(_, d), do: @ll_dims[d] || "d#{d}"

  def format_pred(nil, _env), do: "∅"
  def format_pred(:truep, _env), do: "⊤"
  def format_pred(:falsep, _env), do: "⊥"
  def format_pred({:feat, ["axis", d, t]}, env), do: "#{dim_name(env, d)}<#{t}"
  def format_pred({:feat, ["diag", i, j, c]}, env), do: "#{c}·#{dim_name(env, i)}+#{dim_name(env, j)}<0"
  def format_pred({:feat, ["sq_diag", i, j, c]}, env), do: "#{c}·#{dim_name(env, i)}²+#{dim_name(env, j)}<0"
  def format_pred({:feat, ["prod", i, j, t]}, env), do: "#{dim_name(env, i)}·#{dim_name(env, j)}<#{t}"
  def format_pred({:not, p}, env), do: "¬(#{format_pred(p, env)})"
  def format_pred({:and, p, q}, env), do: "(#{format_pred(p, env)} ∧ #{format_pred(q, env)})"
  def format_pred({:or, p, q}, env), do: "(#{format_pred(p, env)} ∨ #{format_pred(q, env)})"
  def format_pred(other, _env), do: inspect(other)

  @pend_py %{0 => "cos_theta", 1 => "sin_theta", 2 => "omega"}
  @ll_py %{0 => "x", 1 => "y", 2 => "vx", 3 => "vy", 4 => "theta", 5 => "omega"}
  @pong_py %{0 => "ball_x", 1 => "ball_y", 2 => "player_y", 3 => "ball_vx", 4 => "ball_vy", 5 => "enemy_y"}
  @cp_py %{0 => "x", 1 => "x_dot", 2 => "theta", 3 => "theta_dot"}
  @acro_py %{0 => "cos1", 1 => "sin1", 2 => "cos2", 3 => "sin2", 4 => "w1", 5 => "w2"}
  @mc_py %{0 => "pos", 1 => "vel"}
  @bo_py %{0 => "ball_x", 1 => "ball_y", 2 => "paddle_x", 3 => "ball_dx", 4 => "ball_dy"}

  defp dim_py(:pendulum, d), do: @pend_py[d] || "obs[#{d}]"
  defp dim_py(:lunarlander, d), do: @ll_py[d] || "obs[#{d}]"
  defp dim_py(:pong, d), do: @pong_py[d] || "obs[#{d}]"
  defp dim_py(:cartpole, d), do: @cp_py[d] || "obs[#{d}]"
  defp dim_py(:acrobot, d), do: @acro_py[d] || "obs[#{d}]"
  defp dim_py(:mountaincar, d), do: @mc_py[d] || "obs[#{d}]"
  defp dim_py(:breakout, d), do: @bo_py[d] || "obs[#{d}]"
  defp dim_py(_, d), do: "obs[#{d}]"

  defp print_partition_summary(partition, default, action_names, val_seeds, env, max_steps) do
    {val_reward, val_count} = validate_partition(partition, default, action_names, val_seeds, env, max_steps)
    n_regions = length(partition) + 1

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  SYNTHESIS COMPLETE")
    IO.puts("  #{n_regions} regions (#{length(partition)} predicates + default)")
    IO.puts("  Validation: reward=#{Float.round(val_reward / max(length(val_seeds), 1), 1)} avg, successes=#{val_count}/#{length(val_seeds)}")
    IO.puts("══════════════════════════════════════════════════════")

    IO.puts("\nPartition:")
    Enum.each(partition, fn {pred, ranking} ->
      named = Enum.map(ranking, &Map.get(action_names, &1))
      IO.puts("  if #{format_pred(pred, env)}:")
      IO.puts("    → #{format_ranking(named)}")
    end)
    default_named = Enum.map(default, &Map.get(action_names, &1))
    IO.puts("  default:")
    IO.puts("    → #{format_ranking(default_named)}")

    IO.puts("\n=== DEPLOYABLE POLICY ===")
    obs_line = case env do
      :pendulum -> "    cos_theta, sin_theta, omega = obs[:3]"
      :pong -> "    ball_x, ball_y, player_y, ball_vx, ball_vy, enemy_y = obs[:6]"
      :cartpole -> "    x, x_dot, theta, theta_dot = obs[:4]"
      :acrobot -> "    cos1, sin1, cos2, sin2, w1, w2 = obs[:6]"
      :mountaincar -> "    pos, vel = obs[:2]"
      :breakout -> "    ball_x, ball_y, paddle_x, ball_dx, ball_dy = obs[:5]"
      _ -> "    x, y, vx, vy, theta, omega = obs[:6]"
    end
    IO.puts("def policy(obs):")
    IO.puts(obs_line)
    Enum.each(partition, fn {pred, ranking} ->
      named = Enum.map(ranking, &Map.get(action_names, &1))
      IO.puts("    if #{fmt_py(pred, env)}:")
      IO.puts("        return #{inspect(hd(named))}  # #{format_ranking(named)}")
    end)
    IO.puts("    return #{inspect(Map.get(action_names, hd(default)))}  # #{format_ranking(default_named)}")

    IO.puts("\n=== FOR AGDA VERIFICATION ===")
    Enum.each(partition, fn {pred, ranking} ->
      named = Enum.map(ranking, &Map.get(action_names, &1))
      IO.puts("  Region: #{format_pred(pred, env)}")
      IO.puts("    Ranking: #{format_ranking(named)}")
    end)
    IO.puts("  Default:")
    IO.puts("    Ranking: #{format_ranking(default_named)}")
  end

  defp fmt_py({:feat, ["axis", d, t]}, env), do: "#{dim_py(env, d)} < #{t}"
  defp fmt_py({:feat, ["diag", i, j, c]}, env), do: "#{c}*#{dim_py(env, i)} + #{dim_py(env, j)} < 0"
  defp fmt_py({:feat, ["sq_diag", i, j, c]}, env), do: "#{c}*#{dim_py(env, i)}**2 + #{dim_py(env, j)} < 0"
  defp fmt_py({:feat, ["prod", i, j, t]}, env), do: "#{dim_py(env, i)}*#{dim_py(env, j)} < #{t}"
  defp fmt_py({:not, p}, env), do: "not (#{fmt_py(p, env)})"
  defp fmt_py({:and, p, q}, env), do: "(#{fmt_py(p, env)}) and (#{fmt_py(q, env)})"
  defp fmt_py({:or, p, q}, env), do: "(#{fmt_py(p, env)}) or (#{fmt_py(q, env)})"
end
