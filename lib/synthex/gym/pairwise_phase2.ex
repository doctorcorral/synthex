defmodule Synthex.Gym.PairwisePhase2 do
  @moduledoc """
  Propagation-grounded Phase 2: pairwise predicate synthesis.

  For each pair of actions (a,b), collects oracle observations
  (state, a_wins?) from the deployed policy, then uses CEGIS to
  synthesize a PredProg that determines the pairwise ranking.

  By the Propagation Theorem, a PredProg synthesized from
  representative observations generalizes to all feature-equivalent
  states. By Cross-Pair Propagation, a predicate found for one pair
  may automatically determine other pairs.

  The full ranking per region is derived from the pairwise predicates,
  not searched empirically.
  """

  alias Synthex.Gym.Oracle, as: GymOracle
  alias Synthex.Core.{CEGIS, PredProg}

  @pairwise_script "scripts/pairwise_oracle.py"

  def synthesize(chain, default_action, opts) do
    env = Keyword.fetch!(opts, :env)
    actions = Keyword.fetch!(opts, :actions)
    n_actions = length(actions)
    action_names = actions |> Enum.with_index() |> Map.new(fn {a, i} -> {i, a} end)

    seeds = Keyword.get(opts, :seeds, Enum.to_list(0..99))
    max_steps = Keyword.get(opts, :max_steps, nil)
    sample_interval = Keyword.get(opts, :sample_interval, 10)
    depth = Keyword.get(opts, :depth, 0)
    max_coeff = Keyword.get(opts, :max_coeff, 5)

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  Pairwise Predicate Synthesis (Propagation Phase 2)")
    IO.puts("  Env: #{env}, #{length(seeds)} episodes")
    IO.puts("  #{n_actions} actions → #{div(n_actions * (n_actions - 1), 2)} pairs")
    IO.puts("══════════════════════════════════════════════════════\n")

    # Build initial partition from chain
    initial_partition =
      Enum.map(chain, fn {pred, top_action} ->
        top_idx = GymOracle.serialize_action(top_action, env)
        rest = Enum.to_list(0..(n_actions - 1)) -- [top_idx]
        {pred, [top_idx | rest]}
      end)

    default_top_idx = GymOracle.serialize_action(default_action, env)
    default_rest = Enum.to_list(0..(n_actions - 1)) -- [default_top_idx]
    default_ranking = [default_top_idx | default_rest]

    IO.puts("  Phase 1 chain:")
    initial_partition
    |> Enum.with_index()
    |> Enum.each(fn {{pred, ranking}, i} ->
      named = Enum.map(ranking, &Map.get(action_names, &1))
      IO.puts("    R#{i}: #{format_pred(pred, env)} → #{inspect(named)}")
    end)
    named_def = Enum.map(default_ranking, &Map.get(action_names, &1))
    IO.puts("    Default: #{inspect(named_def)}")

    # Step 1: Collect pairwise oracle observations
    IO.puts("\n  Step 1: Collecting pairwise observations...")
    t0 = System.monotonic_time(:millisecond)

    serialized_preds = Enum.map(initial_partition, fn {pred, ranking} ->
      %{"pred" => GymOracle.serialize_pred(pred), "ranking" => ranking}
    end)

    request = %{
      "cmd" => "pairwise_compare",
      "env" => Atom.to_string(env),
      "preds" => serialized_preds,
      "default_ranking" => default_ranking,
      "seeds" => seeds,
      "sample_interval" => sample_interval
    }
    request = if max_steps, do: Map.put(request, "max_steps", max_steps), else: request

    result = call_python(request)

    t1 = System.monotonic_time(:millisecond)
    observations = result["observations"]
    IO.puts("  #{length(observations)} pairwise observations in #{t1 - t0}ms")

    # Step 2: Group observations by (region, pair)
    IO.puts("\n  Step 2: Grouping observations by region and pair...")

    n_regions = length(initial_partition)
    all_pairs = for a <- 0..(n_actions - 2), b <- (a + 1)..(n_actions - 1), do: {a, b}

    grouped = group_observations(observations, n_regions, all_pairs)

    for region_id <- Enum.to_list(0..(n_regions - 1)) ++ [-1] do
      label = if region_id == -1, do: "Default", else: "R#{region_id}"
      pair_counts =
        all_pairs
        |> Enum.map(fn pair ->
          obs = Map.get(grouped, {region_id, pair}, [])
          n_true = Enum.count(obs, fn {_s, b} -> b end)
          n_false = length(obs) - n_true
          {pair, length(obs), n_true, n_false}
        end)
      IO.puts("    #{label}:")
      for {{a, b}, total, n_t, n_f} <- pair_counts do
        na = Map.get(action_names, a)
        nb = Map.get(action_names, b)
        IO.puts("      #{na} vs #{nb}: #{total} obs (#{na}>#{nb}: #{n_t}, #{nb}>#{na}: #{n_f})")
      end
    end

    # Step 3: Collect trajectory states for feature generation
    IO.puts("\n  Step 3: Generating feature space for CEGIS...")
    all_states = Enum.map(observations, fn obs -> obs["state"] end) |> Enum.uniq()
    features = GymOracle.generate_features(all_states, env: env, max_coeff: max_coeff)
    IO.puts("    #{length(features)} features from #{length(all_states)} unique states")

    # Step 4: CEGIS per pair — synthesize predicates
    IO.puts("\n  Step 4: CEGIS predicate synthesis per pair...")
    IO.puts("    Version space depth: #{depth}")

    eval_feat_fn = fn feature, state ->
      GymOracle.eval_pred({:feat, feature}, state)
    end

    pair_predicates =
      for region_id <- Enum.to_list(0..(n_regions - 1)) ++ [-1] do
        label = if region_id == -1, do: "Default", else: "R#{region_id}"

        pair_results =
          for {a, b} = pair <- all_pairs do
            obs = Map.get(grouped, {region_id, pair}, [])
            na = Map.get(action_names, a)
            nb = Map.get(action_names, b)

            if length(obs) < 3 do
              IO.puts("    #{label} #{na}>#{nb}: too few observations (#{length(obs)})")
              {pair, :insufficient, nil, 0.0}
            else
              n_true = Enum.count(obs, fn {_s, b} -> b end)
              ratio = n_true / length(obs)

              if ratio > 0.95 or ratio < 0.05 do
                winner = if ratio > 0.5, do: a, else: b
                wname = Map.get(action_names, winner)
                IO.puts("    #{label} #{na} vs #{nb}: unanimous → #{wname} always wins (#{Float.round(ratio * 100, 1)}%)")
                {pair, :unanimous, winner, ratio}
              else
                pred_obs = Enum.map(obs, fn {state, bool} -> {state, bool} end)

                result = progressive_cegis(features, pred_obs, eval_feat_fn, depth)

                case result do
                  {:exhausted, best_depth} ->
                    IO.puts("    #{label} #{na} vs #{nb}: CEGIS exhausted through depth #{best_depth}")
                    {pair, :exhausted, nil, ratio}

                  {:found, best, n_survivors, at_depth} ->
                    n_match = Enum.count(pred_obs, fn {state, target} ->
                      PredProg.eval(best, state, eval_feat_fn) == target
                    end)
                    accuracy = n_match / length(pred_obs)

                    IO.puts("    #{label} #{na} vs #{nb}: #{n_survivors} survivors at depth #{at_depth}, " <>
                            "best=#{format_predprog(best)}, accuracy=#{Float.round(accuracy * 100, 1)}%")

                    cross_matches = check_cross_pair(best, grouped, region_id, pair, all_pairs, eval_feat_fn)
                    if length(cross_matches) > 0 do
                      cross_str = Enum.map(cross_matches, fn {{ca, cb}, acc} ->
                        "#{Map.get(action_names, ca)}>#{Map.get(action_names, cb)}(#{Float.round(acc * 100, 1)}%)"
                      end) |> Enum.join(", ")
                      IO.puts("      Cross-pair propagation: #{cross_str}")
                    end

                    {pair, :synthesized, best, accuracy}
                end
              end
            end
          end

        {region_id, pair_results}
      end

    # Step 5: Derive full rankings from pairwise predicates
    IO.puts("\n  Step 5: Deriving full rankings from pairwise structure...")

    final_rankings =
      for {region_id, pair_results} <- pair_predicates do
        label = if region_id == -1, do: "Default", else: "R#{region_id}"

        wins = build_pairwise_matrix(pair_results, all_pairs, n_actions,
                                     all_states, region_id, initial_partition, eval_feat_fn)

        win_counts =
          for a <- 0..(n_actions - 1) do
            count = Enum.count(0..(n_actions - 1), fn b ->
              a != b and Map.get(wins, {a, b}, false)
            end)
            {a, count}
          end
          |> Enum.sort_by(fn {_a, c} -> -c end)

        ranking = Enum.map(win_counts, fn {a, _c} -> a end)
        named = Enum.map(ranking, &Map.get(action_names, &1))

        IO.puts("    #{label}: #{format_named_ranking(named)}")
        win_counts
        |> Enum.each(fn {a, c} ->
          IO.puts("      #{Map.get(action_names, a)}: #{c} wins")
        end)

        {region_id, ranking}
      end

    # Build final partition
    final_pred_rankings =
      chain
      |> Enum.with_index()
      |> Enum.map(fn {{pred, _top}, i} ->
        {_rid, ranking} = Enum.find(final_rankings, fn {rid, _} -> rid == i end)
        {pred, ranking}
      end)

    {_def_id, final_default} = Enum.find(final_rankings, fn {rid, _} -> rid == -1 end)

    # Step 6: Validate
    IO.puts("\n  Step 6: Validating on 500 held-out seeds...")
    val_seeds = Enum.to_list(10_000..10_499)
    ms = max_steps || default_max_steps(env)
    {val_reward, val_wins} =
      validate_partition(final_pred_rankings, final_default, action_names,
                         val_seeds, env, ms)
    val_avg = Float.round(val_reward / length(val_seeds), 1)
    IO.puts("  Validation: #{val_wins}/#{length(val_seeds)} wins, avg reward=#{val_avg}")

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  COMPLETE")
    IO.puts("══════════════════════════════════════════════════════\n")

    {final_pred_rankings, final_default}
  end

  # ── Progressive CEGIS ──────────────────────────────────────

  defp progressive_cegis(features, observations, eval_feat_fn, max_depth) do
    atoms = CEGIS.enumerate(features, 0)
    survivors_0 = CEGIS.refine(atoms, observations, eval_feat_fn)

    if length(survivors_0) > 0 do
      {:found, hd(survivors_0), length(survivors_0), 0}
    else
      if max_depth < 1, do: {:exhausted, 0}, else: try_depth_1(features, observations, eval_feat_fn)
    end
  end

  defp try_depth_1(features, observations, eval_feat_fn) do
    positives = for {s, true} <- observations, do: s
    negatives = for {s, false} <- observations, do: s

    discriminative =
      features
      |> Task.async_stream(fn feat ->
        p_true = Enum.count(positives, fn s -> eval_feat_fn.(feat, s) end)
        p_false = length(positives) - p_true
        n_true = Enum.count(negatives, fn s -> eval_feat_fn.(feat, s) end)
        n_false = length(negatives) - n_true

        useful = p_true > 0 and p_false > 0 or n_true > 0 and n_false > 0
        {feat, useful}
      end, ordered: false)
      |> Enum.filter(fn {:ok, {_f, useful}} -> useful end)
      |> Enum.map(fn {:ok, {f, _}} -> f end)

    IO.puts("    Depth-1 filter: #{length(discriminative)}/#{length(features)} discriminative features")

    if length(discriminative) == 0 do
      {:exhausted, 1}
    else
      disc_atoms = Enum.map(discriminative, fn f -> {:feat, f} end)

      negations = Enum.map(disc_atoms, fn p -> {:not, p} end)
      ands = for p1 <- disc_atoms, p2 <- disc_atoms, p1 != p2, do: {:and, p1, p2}
      neg_ands = for p1 <- negations, p2 <- disc_atoms, do: {:and, p1, p2}

      depth1 = disc_atoms ++ negations ++ ands ++ neg_ands
      IO.puts("    Depth-1 version space: #{length(depth1)} programs")

      survivors_1 = CEGIS.refine(depth1, observations, eval_feat_fn)

      if length(survivors_1) > 0 do
        {:found, hd(survivors_1), length(survivors_1), 1}
      else
        {:exhausted, 1}
      end
    end
  end

  # ── Helpers ──────────────────────────────────────────────────

  defp group_observations(observations, _n_regions, _all_pairs) do
    Enum.reduce(observations, %{}, fn obs, acc ->
      region = obs["region"]
      [a, b] = obs["pair"]
      pair = {a, b}
      state = obs["state"]
      a_wins = obs["a_wins"]
      key = {region, pair}
      Map.update(acc, key, [{state, a_wins}], fn existing ->
        [{state, a_wins} | existing]
      end)
    end)
  end

  defp check_cross_pair(predicate, grouped, region_id, source_pair,
                         all_pairs, eval_feat_fn) do
    other_pairs = all_pairs -- [source_pair]

    for pair <- other_pairs,
        obs = Map.get(grouped, {region_id, pair}, []),
        length(obs) >= 3 do
      n_match = Enum.count(obs, fn {state, target} ->
        PredProg.eval(predicate, state, eval_feat_fn) == target
      end)
      accuracy = n_match / length(obs)
      if accuracy > 0.8, do: {pair, accuracy}, else: nil
    end
    |> Enum.reject(&is_nil/1)
  end

  defp build_pairwise_matrix(pair_results, _all_pairs, _n_actions,
                              all_states, region_id, initial_partition,
                              eval_feat_fn) do
    region_states =
      all_states
      |> Enum.filter(fn s ->
        cond do
          region_id == -1 ->
            not Enum.any?(initial_partition, fn {pred, _} ->
              GymOracle.eval_pred(pred, s)
            end)
          true ->
            {pred, _} = Enum.at(initial_partition, region_id)
            GymOracle.eval_pred(pred, s)
        end
      end)
      |> Enum.take(100)

    for {pair, status, value, _ratio} <- pair_results, into: %{} do
      {a, b} = pair
      case status do
        :unanimous ->
          if value == a do
            {{a, b}, true}
          else
            {{b, a}, true}
          end

        :synthesized ->
          pred = value
          n_true = Enum.count(region_states, fn s ->
            PredProg.eval(pred, s, eval_feat_fn)
          end)
          if n_true >= length(region_states) / 2 do
            {{a, b}, true}
          else
            {{b, a}, true}
          end

        _ ->
          {{a, b}, false}
      end
    end
  end

  # ── Validation ──────────────────────────────────────────────

  defp validate_partition(partition, default, action_names, seeds, env, max_steps) do
    chain = Enum.map(partition, fn {pred, ranking} ->
      named_top = Map.get(action_names, hd(ranking))
      {pred, named_top}
    end)
    serialized = GymOracle.serialize_chain(chain, env)
    default_named_top = Map.get(action_names, hd(default))
    default_int = GymOracle.serialize_action(default_named_top, env)

    script = GymOracle.oracle_script(env)
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

    result = call_python_env(request, script)
    {result["baseline_reward"], result["baseline_landings"] || 0}
  end

  defp default_max_steps(:lunarlander), do: 1000
  defp default_max_steps(:pong), do: 10_000
  defp default_max_steps(:breakout), do: 10_000
  defp default_max_steps(:cartpole), do: 500
  defp default_max_steps(:acrobot), do: 500
  defp default_max_steps(:mountaincar), do: 200
  defp default_max_steps(_), do: 1000

  # ── Python bridge ──────────────────────────────────────────

  defp call_python(request) do
    script = @pairwise_script
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_pw_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_pw_resp_#{uid}.json")

    File.write!(req_file, Jason.encode!(request))

    {output, exit_code} =
      System.cmd("python3", ["-u", script, req_file, resp_file],
        stderr_to_stdout: true,
        cd: Path.expand("../../..", __DIR__)
      )

    if exit_code != 0 do
      IO.puts("  [PairwisePhase2] Python error (code #{exit_code})")
      IO.puts("  #{String.slice(output, 0, 500)}")
    end

    result = Jason.decode!(File.read!(resp_file))
    File.rm(req_file)
    File.rm(resp_file)
    result
  end

  defp call_python_env(request, script) do
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_pw_v_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_pw_v_resp_#{uid}.json")

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

  # ── Formatting ─────────────────────────────────────────────

  defp format_pred(pred, env) do
    Synthex.Gym.Ranking.format_pred(pred, env)
  rescue
    _ -> inspect(pred)
  end

  defp format_predprog(:truep), do: "truep"
  defp format_predprog(:falsep), do: "falsep"
  defp format_predprog({:feat, f}), do: inspect(f)
  defp format_predprog({:not, p}), do: "¬(#{format_predprog(p)})"
  defp format_predprog({:and, p, q}), do: "(#{format_predprog(p)} ∧ #{format_predprog(q)})"
  defp format_predprog({:or, p, q}), do: "(#{format_predprog(p)} ∨ #{format_predprog(q)})"

  defp format_named_ranking(named_list) do
    named_list
    |> Enum.map(&to_string/1)
    |> Enum.join(" > ")
  end

end
