defmodule Synthex.Gym.PerStateRanking do
  @moduledoc """
  Per-state episode-reward ranking with region-level aggregation
  for CoindHomo-principled Phase 2.

  Profiles the reward of each action at individual states along the
  deployed policy's trajectory, then aggregates at the region level:

    1. Deploy the chain policy (from Phase 1 coordinate descent)
    2. At each sampled state s along the trajectory:
       - Clone/replay the environment state
       - Try each action a, then follow the deployed policy
       - Record the cumulative episode reward for each action
    3. For each region R, sum the per-action rewards across all
       sampled states in R, then rank by total reward
    4. Report consistency: % of states where the per-state ranking
       agrees with the aggregated region ranking

  The aggregation smooths out per-state variance while preserving
  the per-state grounding: the ranking emerges from many branching
  decisions under the policy's own dynamics, which is what the
  CoindHomo self-consistency condition requires.
  """

  alias Synthex.Gym.Oracle, as: GymOracle

  @per_state_script "scripts/per_state_oracle.py"

  @doc """
  Run the principled Phase 2 on a chain from Phase 1.

  Takes the chain (list of {pred, top_action} tuples), default action,
  and produces full rankings per region via per-state profiling.

  Options:
    - env: environment atom (:lunarlander, :pong, etc.)
    - actions: list of action atoms
    - seeds: episode seeds for profiling
    - max_steps: max steps per episode
    - sample_interval: profile every N steps (default 10)
  """
  def determine_rankings(chain, default_action, opts) do
    env = Keyword.fetch!(opts, :env)
    actions = Keyword.fetch!(opts, :actions)
    n_actions = length(actions)
    action_names = actions |> Enum.with_index() |> Map.new(fn {a, i} -> {i, a} end)

    seeds = Keyword.get(opts, :seeds, Enum.to_list(0..49))
    max_steps = Keyword.get(opts, :max_steps, nil)
    sample_interval = Keyword.get(opts, :sample_interval, 10)

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  Per-State Episode-Reward Ranking (CoindHomo Phase 2)")
    IO.puts("  Env: #{env}, #{length(seeds)} episodes, interval=#{sample_interval}")
    IO.puts("══════════════════════════════════════════════════════\n")

    initial_partition =
      Enum.map(chain, fn {pred, top_action} ->
        top_idx = GymOracle.serialize_action(top_action, env)
        rest = Enum.to_list(0..(n_actions - 1)) -- [top_idx]
        {pred, [top_idx | rest]}
      end)

    default_top_idx = GymOracle.serialize_action(default_action, env)
    default_rest = Enum.to_list(0..(n_actions - 1)) -- [default_top_idx]
    default_ranking = [default_top_idx | default_rest]

    IO.puts("  Initial partition (top action from Phase 1):")
    initial_partition
    |> Enum.with_index()
    |> Enum.each(fn {{pred, ranking}, i} ->
      named = Enum.map(ranking, &Map.get(action_names, &1))
      IO.puts("    R#{i}: #{format_pred(pred, env)} → #{inspect(named)}")
    end)
    named_def = Enum.map(default_ranking, &Map.get(action_names, &1))
    IO.puts("    Default: #{inspect(named_def)}")

    IO.puts("\n  Profiling per-state rankings...")
    t0 = System.monotonic_time(:millisecond)

    serialized_preds = Enum.map(initial_partition, fn {pred, ranking} ->
      %{"pred" => GymOracle.serialize_pred(pred), "ranking" => ranking}
    end)

    request = %{
      "cmd" => "per_state_rank",
      "env" => Atom.to_string(env),
      "preds" => serialized_preds,
      "default_ranking" => default_ranking,
      "seeds" => seeds,
      "sample_interval" => sample_interval
    }
    request = if max_steps, do: Map.put(request, "max_steps", max_steps), else: request

    result = call_python(request)

    t1 = System.monotonic_time(:millisecond)
    n_points = result["n_profile_points"]
    IO.puts("  Done: #{n_points} profile points in #{t1 - t0}ms\n")

    analysis = result["analysis"]

    IO.puts("  ┌─────────────────────────────────────────────────")
    IO.puts("  │ Region-Aggregated Rankings (sum of per-state rewards)")
    IO.puts("  ├─────────────────────────────────────────────────")

    n_regions = length(initial_partition)
    all_regions = Enum.to_list(0..(n_regions - 1)) ++ [-1]

    final_partition =
      Enum.map(all_regions, fn region_id ->
        key = Integer.to_string(region_id)
        region_data = analysis[key] || %{"n_states" => 0}
        n_states = region_data["n_states"]
        ranking = region_data["ranking"]
        avg_rewards = region_data["avg_rewards"] || []
        consistency = region_data["consistency"]
        top_consistency = region_data["top_action_consistency"]

        region_label =
          if region_id == -1, do: "Default",
          else: "R#{region_id}"

        if n_states == 0 do
          IO.puts("  │ #{region_label}: no states visited")
        else
          named_ranking = if ranking do
            Enum.map(ranking, &Map.get(action_names, &1))
          else
            []
          end

          IO.puts("  │ #{region_label}: #{n_states} states")
          IO.puts("  │   Avg rewards per action:")
          avg_rewards
          |> Enum.with_index()
          |> Enum.each(fn {avg, idx} ->
            IO.puts("  │     #{Map.get(action_names, idx)}: #{avg}")
          end)
          IO.puts("  │   Ranking: #{format_ranking(named_ranking)}")
          IO.puts("  │   Top-action agreement: #{top_consistency}%")
          IO.puts("  │   Full-ranking agreement: #{consistency}%")
        end

        {region_id, ranking}
      end)

    IO.puts("  └─────────────────────────────────────────────────\n")

    final_pred_rankings =
      initial_partition
      |> Enum.with_index()
      |> Enum.map(fn {{pred, _initial_ranking}, i} ->
        {_region_id, ranking} =
          Enum.find(final_partition, fn {rid, _} -> rid == i end)
        {pred, ranking || Enum.to_list(0..(n_actions - 1))}
      end)

    {_default_id, default_agg} =
      Enum.find(final_partition, fn {rid, _} -> rid == -1 end)
    final_default = default_agg || default_ranking

    IO.puts("  Final full-ranking partition:")
    final_pred_rankings
    |> Enum.with_index()
    |> Enum.each(fn {{pred, ranking}, i} ->
      named = Enum.map(ranking, &Map.get(action_names, &1))
      IO.puts("    R#{i}: #{format_pred(pred, env)} → #{format_ranking(named)}")
    end)
    named_final_def = Enum.map(final_default, &Map.get(action_names, &1))
    IO.puts("    Default: #{format_ranking(named_final_def)}")

    IO.puts("\n  Validating final partition on 500 held-out seeds...")
    val_seeds = Enum.to_list(10_000..10_499)
    {val_reward, val_wins} =
      validate_partition(final_pred_rankings, final_default, action_names,
                         val_seeds, env, max_steps || default_max_steps(env))
    val_avg = Float.round(val_reward / length(val_seeds), 1)
    IO.puts("  Validation: #{val_wins}/#{length(val_seeds)} wins, avg reward=#{val_avg}")

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  COMPLETE")
    IO.puts("══════════════════════════════════════════════════════\n")

    {final_pred_rankings, final_default}
  end

  @doc """
  Exhaustive Phase 2: fix top actions from Phase 1, enumerate ALL orderings
  of remaining actions across all regions simultaneously, evaluate each
  complete policy on episodes, find the best.

  With k actions and r regions this is ((k-1)!)^r candidates.
  E.g. 4 actions, 4 regions → 6^4 = 1,296 policies.
  """
  def exhaustive_rankings(chain, default_action, opts) do
    env = Keyword.fetch!(opts, :env)
    actions = Keyword.fetch!(opts, :actions)
    n_actions = length(actions)
    action_names = actions |> Enum.with_index() |> Map.new(fn {a, i} -> {i, a} end)

    seeds = Keyword.get(opts, :seeds, Enum.to_list(0..99))
    max_steps = Keyword.get(opts, :max_steps, nil)

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  Exhaustive Full-Ranking Search")
    IO.puts("  Env: #{env}, #{length(seeds)} episodes")
    IO.puts("══════════════════════════════════════════════════════\n")

    preds_serialized = Enum.map(chain, fn {pred, _top} ->
      %{"pred" => GymOracle.serialize_pred(pred)}
    end)

    top_actions =
      Enum.map(chain, fn {_pred, top} ->
        GymOracle.serialize_action(top, env)
      end) ++ [GymOracle.serialize_action(default_action, env)]

    IO.puts("  Top actions (from Phase 1):")
    chain
    |> Enum.with_index()
    |> Enum.each(fn {{pred, top}, i} ->
      IO.puts("    R#{i}: #{format_pred(pred, env)} → #{top}")
    end)
    IO.puts("    Default: #{default_action}\n")

    n_regions = length(chain) + 1
    n_candidates = :math.pow(factorial(n_actions - 1), n_regions) |> round()
    IO.puts("  Enumerating #{n_candidates} candidate policies...\n")

    t0 = System.monotonic_time(:millisecond)

    request = %{
      "cmd" => "exhaustive_rank",
      "env" => Atom.to_string(env),
      "preds" => preds_serialized,
      "top_actions" => top_actions,
      "n_actions" => n_actions,
      "seeds" => seeds
    }
    request = if max_steps, do: Map.put(request, "max_steps", max_steps), else: request

    result = call_python(request)

    t1 = System.monotonic_time(:millisecond)
    n_eval = result["n_evaluated"]
    IO.puts("  Evaluated #{n_eval} policies in #{t1 - t0}ms\n")

    top_policies = result["top_policies"]

    IO.puts("  ┌─────────────────────────────────────────────────")
    IO.puts("  │ Top 10 Policies (by avg episode reward)")
    IO.puts("  ├─────────────────────────────────────────────────")

    top_policies
    |> Enum.take(10)
    |> Enum.with_index(1)
    |> Enum.each(fn {policy, rank} ->
      rankings = policy["policy"]
      avg = policy["avg_reward"]
      wins = policy["wins"]

      region_strs =
        rankings
        |> Enum.with_index()
        |> Enum.map(fn {ranking, i} ->
          named = Enum.map(ranking, &Map.get(action_names, &1))
          label = if i < length(chain), do: "R#{i}", else: "Def"
          "#{label}=#{format_ranking(named)}"
        end)
        |> Enum.join(", ")

      IO.puts("  │ ##{rank}: avg=#{avg}, wins=#{wins}/#{length(seeds)}")
      IO.puts("  │      #{region_strs}")
    end)

    IO.puts("  └─────────────────────────────────────────────────\n")

    best = hd(top_policies)
    best_rankings = best["policy"]

    best_pred_rankings =
      chain
      |> Enum.with_index()
      |> Enum.map(fn {{pred, _top}, i} ->
        {pred, Enum.at(best_rankings, i)}
      end)

    best_default = List.last(best_rankings)

    IO.puts("  Best policy:")
    best_pred_rankings
    |> Enum.with_index()
    |> Enum.each(fn {{pred, ranking}, i} ->
      named = Enum.map(ranking, &Map.get(action_names, &1))
      IO.puts("    R#{i}: #{format_pred(pred, env)} → #{format_ranking(named)}")
    end)
    named_def = Enum.map(best_default, &Map.get(action_names, &1))
    IO.puts("    Default: #{format_ranking(named_def)}")

    IO.puts("\n  Validating on 500 held-out seeds...")
    val_seeds = Enum.to_list(10_000..10_499)
    {val_reward, val_wins} =
      validate_partition(best_pred_rankings, best_default, action_names,
                         val_seeds, env, max_steps || default_max_steps(env))
    val_avg = Float.round(val_reward / length(val_seeds), 1)
    IO.puts("  Validation: #{val_wins}/#{length(val_seeds)} wins, avg reward=#{val_avg}")

    IO.puts("\n══════════════════════════════════════════════════════")
    IO.puts("  COMPLETE")
    IO.puts("══════════════════════════════════════════════════════\n")

    {best_pred_rankings, best_default}
  end

  defp factorial(0), do: 1
  defp factorial(n) when n > 0, do: n * factorial(n - 1)

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
    script = @per_state_script
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_psr_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_psr_resp_#{uid}.json")

    File.write!(req_file, Jason.encode!(request))

    {output, exit_code} =
      System.cmd("python3", ["-u", script, req_file, resp_file],
        stderr_to_stdout: true,
        cd: Path.expand("../../..", __DIR__)
      )

    if exit_code != 0 do
      IO.puts("  [PerStateRanking] Python exited with code #{exit_code}")
      IO.puts("  Output: #{String.slice(output, 0, 500)}")
    end

    result = Jason.decode!(File.read!(resp_file))
    File.rm(req_file)
    File.rm(resp_file)
    result
  end

  defp call_python_env(request, script) do
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_psr_v_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_psr_v_resp_#{uid}.json")

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

  defp format_ranking(named_list) do
    named_list
    |> Enum.map(&to_string/1)
    |> Enum.join(" > ")
  end
end
