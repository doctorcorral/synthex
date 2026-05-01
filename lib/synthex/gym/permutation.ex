defmodule Synthex.Gym.Permutation do
  @moduledoc """
  Predicate-guided action ranking synthesis for regulation tasks.

  A predicate p partitions the continuous state space into two regions.
  Each region is assigned a complete ranking (total order) over all actions.
  Grounded in the CSHRL coinductive framework.
  """

  alias Synthex.Gym.Oracle, as: GymOracle
  alias Synthex.Core.CEGIS

  def solve(actions, opts \\ []) do
    env = Keyword.fetch!(opts, :env)
    max_steps = Keyword.fetch!(opts, :max_steps)
    depth = Keyword.get(opts, :depth, 1)
    max_coeff = Keyword.get(opts, :max_coeff, 5)
    n_episodes = Keyword.get(opts, :n_episodes, 200)
    top_k = Keyword.get(opts, :top_k, 30)
    cegar_rounds = Keyword.get(opts, :cegar_rounds, 3)
    _top_pairs = Keyword.get(opts, :top_pairs, 3)

    all_rankings = permutations(actions)
    ranking_pairs = for r1 <- all_rankings, r2 <- all_rankings,
                        hd(r1) != hd(r2), do: {r1, r2}
    val_seeds = Enum.to_list(10_000..10_499)

    IO.puts("  Ranking Synthesis -- Single Predicate")
    IO.puts("  Env: #{env}, Actions: #{inspect(actions)}")
    IO.puts("  Rankings: #{length(all_rankings)}, Pairs: #{length(ranking_pairs)}")
    IO.puts("  Depth: #{depth}, Episodes: #{n_episodes}, max_steps: #{max_steps}\n")

    {states, _} = GymOracle.get_trajectory_states([], hd(actions),
      env: env, seeds: Enum.to_list(0..39), max_steps: max_steps)
    features = GymOracle.generate_features(states, env: env, max_coeff: max_coeff)
    IO.puts("  #{length(features)} initial features\n")

    initial_best = {nil, nil, nil, -999_999.0, 0}

    {_final_features, final_best} =
      Enum.reduce(1..cegar_rounds, {features, initial_best}, fn round, {feats, best} ->
        IO.puts("\n  CEGAR Round #{round}/#{cegar_rounds} -- #{length(feats)} features")

        seed_offset = (round - 1) * n_episodes
        seeds = Enum.to_list(seed_offset..(seed_offset + n_episodes - 1))
        atoms = CEGIS.enumerate(feats, 0)

        executable_pairs = ranking_pairs
          |> Enum.uniq_by(fn {r1, r2} -> {hd(r1), hd(r2)} end)

        d0_results =
          Enum.map(executable_pairs, fn {r_true, r_false} ->
            score_ranking_pair(atoms, r_true, r_false, seeds, env, max_steps)
          end)

        {d0_pred, d0_rt, d0_rf, d0_reward, d0_count} = best_from_results(d0_results, atoms)

        {round_pred, round_rt, round_rf, _round_reward, _round_count} =
          if depth >= 1 do
            search_depth_1(d0_results, d0_pred, d0_rt, d0_rf, d0_reward, d0_count,
                           atoms, top_k, seeds, env, max_steps)
          else
            {d0_pred, d0_rt, d0_rf, d0_reward, d0_count}
          end

        {new_best, feats} =
          if round_pred != nil do
            {val_reward, val_count} = validate_policy(round_pred, round_rt, round_rf, val_seeds, env, max_steps)
            IO.puts("  Validation: reward=#{Float.round(val_reward, 1)} successes=#{val_count}/#{length(val_seeds)}")

            {_, _, _, prev_val, _} = best
            updated_best = if val_reward > prev_val do
              IO.puts("  New best!")
              {round_pred, round_rt, round_rf, val_reward, val_count}
            else
              best
            end

            feats = if round < cegar_rounds do
              a_true = hd(round_rt)
              a_false = hd(round_rf)
              chain = [{round_pred, a_true}]
              {new_feats, _, _, _, _} =
                GymOracle.find_counterexamples(chain, a_false, feats,
                  env: env, max_coeff: max_coeff, max_steps: max_steps)
              if length(new_feats) > 0, do: feats ++ new_feats, else: feats
            else
              feats
            end

            {updated_best, feats}
          else
            {best, feats}
          end

        {feats, new_best}
      end)

    {best_pred, best_rt, best_rf, best_val, best_stab} = final_best
    IO.puts("\n  SYNTHESIS COMPLETE")
    IO.puts("  Best: reward=#{Float.round(best_val, 1)} successes=#{best_stab}/#{length(val_seeds)}")
    if best_pred, do: IO.puts("  Pred: #{GymOracle.format_pred(best_pred, env)}")
    {best_pred, best_rt, best_rf}
  end

  defp score_ranking_pair(candidates, r_true, r_false, seeds, env, max_steps) do
    a_true = hd(r_true)
    a_false = hd(r_false)

    {scored, baseline, _} =
      GymOracle.score_candidates(candidates, a_true, a_false, [],
        seeds: seeds, chain_after: [], env: env, max_steps: max_steps)

    best = Enum.max_by(scored, fn {_i, r, _l} -> r end, fn -> nil end)

    case best do
      nil -> {r_true, r_false, scored, baseline, nil, nil}
      {idx, reward, _count} -> {r_true, r_false, scored, baseline, idx, reward}
    end
  end

  defp best_from_results(results, candidates) do
    results
    |> Enum.filter(fn {_, _, _, _, idx, _} -> idx != nil end)
    |> Enum.max_by(fn {_, _, _, _, _, reward} -> reward end, fn -> nil end)
    |> case do
      nil -> {nil, nil, nil, -999_999.0, 0}
      {r_true, r_false, scored, _, idx, reward} ->
        pred = Enum.at(candidates, idx)
        count = case Enum.find(scored, fn {i, _, _} -> i == idx end) do
          {_, _, c} -> c
          _ -> 0
        end
        {pred, r_true, r_false, reward, count}
    end
  end

  defp search_depth_1(d0_results, d0_pred, d0_rt, d0_rf, d0_reward, d0_count,
                       atoms, top_k, seeds, env, max_steps) do
    ranked_pairs =
      d0_results
      |> Enum.filter(fn {_, _, _, _, idx, _} -> idx != nil end)
      |> Enum.sort_by(fn {_, _, _, _, _, r} -> -r end)

    top_atoms =
      ranked_pairs
      |> Enum.flat_map(fn {_, _, scored, _, _, _} -> scored end)
      |> Enum.sort_by(fn {_idx, r, _l} -> -r end)
      |> Enum.uniq_by(fn {idx, _r, _l} -> idx end)
      |> Enum.take(top_k)
      |> Enum.map(fn {idx, _r, _l} -> Enum.at(atoms, idx) end)

    negations = Enum.map(top_atoms, fn p -> {:not, p} end)
    d1_candidates =
      (for p <- top_atoms, q <- top_atoms, p != q, do: {:and, p, q}) ++
      (for p <- top_atoms, q <- top_atoms, p != q, do: {:or, p, q}) ++
      (for p <- negations, q <- top_atoms, do: {:and, p, q}) ++
      (for p <- negations, q <- top_atoms, do: {:or, p, q})
      |> Enum.uniq()

    best_pair = hd(ranked_pairs)
    {bp_rt, bp_rf, _, _, _, _} = best_pair

    d1_results = [score_ranking_pair(d1_candidates, bp_rt, bp_rf, seeds, env, max_steps)]
    {d1_pred, d1_rt, d1_rf, d1_reward, d1_count} = best_from_results(d1_results, d1_candidates)

    if d1_pred != nil and d1_reward > d0_reward do
      {d1_pred, d1_rt, d1_rf, d1_reward, d1_count}
    else
      {d0_pred, d0_rt, d0_rf, d0_reward, d0_count}
    end
  end

  defp validate_policy(pred, r_true, r_false, val_seeds, env, max_steps) do
    a_true = hd(r_true)
    a_false = hd(r_false)
    chain = [{pred, a_true}]
    serialized = GymOracle.serialize_chain(chain, env)
    default_int = GymOracle.serialize_action(a_false, env)

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

  defp call_python(request, env) do
    script = GymOracle.oracle_script(env)
    python = Application.get_env(:synthex, :python, "python3")
    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_rank_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_rank_resp_#{uid}.json")

    File.write!(req_file, Jason.encode!(request))
    {_output, _exit_code} =
      System.cmd(python, ["-u", script, req_file, resp_file],
        stderr_to_stdout: true,
        cd: Application.get_env(:synthex, :project_root, Path.expand("../../..", __DIR__))
      )

    result = Jason.decode!(File.read!(resp_file))
    File.rm(req_file)
    File.rm(resp_file)
    result
  end

  defp permutations([]), do: [[]]
  defp permutations(list) do
    for elem <- list,
        rest <- permutations(list -- [elem]),
        do: [elem | rest]
  end
end
