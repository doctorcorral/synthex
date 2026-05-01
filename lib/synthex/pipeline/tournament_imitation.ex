defmodule Synthex.Pipeline.TournamentImitation do
  @moduledoc """
  Fits a Permutation Tree (RankTree) to a Tournament Oracle.
  This allows us to take independently synthesized pairwise predicates (even partial ones),
  use them to generate a ground-truth dataset via imitation learning,
  and fit a formally verified, cycle-free RankTree.
  """

  alias Synthex.Pure.Permutation.TreePolicy
  alias Synthex.Core.{PredProg, ContinuousFeatures, CEGIS}

  @doc """
  Hardcoded Tournament Oracle based on the successful partial pairwise synthesis for LunarLander.
  """
  def lunarlander_tournament_oracle([x, y, vx, vy, theta, _omega]) do
    m_beats_n = (vy < -1.1) or (6 * y + vy < 0)
    l_beats_n = (x >= -0.5) and (y < 1.374)
    r_beats_n = (x <= 0.5) and (y < 1.374)
    m_beats_l = (y < 0.8768) and (3 * vx + vy < 0)
    m_beats_r = (y < 0.8768) and (-3 * vx + vy < 0)

    best_action = :do_nothing
    best_action = if l_beats_n, do: :fire_left, else: best_action

    best_action = if r_beats_n do
      if best_action == :fire_left do
        if theta > 0 or (theta == 0 and x > 0), do: :fire_right, else: :fire_left
      else
        :fire_right
      end
    else
      best_action
    end

    case best_action do
      :do_nothing -> if m_beats_n, do: :fire_main, else: best_action
      :fire_left  -> if m_beats_l, do: :fire_main, else: best_action
      :fire_right -> if m_beats_r, do: :fire_main, else: best_action
    end
  end

  def lunarlander_oracle_ranking(state_list) do
    best = lunarlander_tournament_oracle(state_list)
    actions = [:do_nothing, :fire_left, :fire_main, :fire_right]
    [best | List.delete(actions, best)]
  end

  def collect_dataset(env_mod, num_episodes, steps_per_episode) do
    IO.puts("Collecting Imitation Dataset from Tournament Oracle...")

    states = Enum.flat_map(1..num_episodes, fn _ ->
      start_state = Enum.random(env_mod.starts())
      rollout_oracle(start_state, env_mod, steps_per_episode)
    end)

    unique_states = Enum.uniq(states)
    IO.puts("Collected #{length(unique_states)} unique states.")

    Enum.map(unique_states, fn s ->
      sl = env_mod.state_to_list(s)
      scaled_sl = Enum.map(sl, &(&1 / 100_000_000.0))
      true_rank = lunarlander_oracle_ranking(scaled_sl)
      {s, sl, true_rank}
    end)
  end

  defp rollout_oracle(state, _env_mod, 0), do: [state]
  defp rollout_oracle(state, env_mod, n) do
    if env_mod.terminal?(state) do
      [state]
    else
      sl = env_mod.state_to_list(state)
      scaled_sl = Enum.map(sl, &(&1 / 100_000_000.0))
      action = lunarlander_tournament_oracle(scaled_sl)
      s_prime = apply_substeps(state, action, env_mod, 5)
      [state | rollout_oracle(s_prime, env_mod, n - 1)]
    end
  end

  defp apply_substeps(state, _action, _env_mod, 0), do: state
  defp apply_substeps(state, action, env_mod, steps_left) do
    if env_mod.terminal?(state) do
      state
    else
      s_prime = env_mod.step(state, action)
      apply_substeps(s_prime, action, env_mod, steps_left - 1)
    end
  end

  def solve() do
    env_mod = Synthex.Envs.LunarLander
    eval_fn = &ContinuousFeatures.eval_feature/2

    dataset = collect_dataset(env_mod, 50, 100)

    traj_lists = Enum.map(dataset, fn {_, sl, _} -> sl end)
    state_size = length(hd(traj_lists))

    max_coeff = 1
    features = ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
    cands = CEGIS.enumerate(features, 0)

    pruned_cands = Enum.filter(cands, fn pred ->
      evals = Enum.map(dataset, fn {_, sl, _} -> PredProg.eval(pred, sl, eval_fn) end)
      Enum.any?(evals, &(&1 == true)) and Enum.any?(evals, &(&1 == false))
    end)

    IO.puts("Generated #{length(pruned_cands)} active predicates.")

    actions = env_mod.actions()
    all_rankings = Synthex.Pure.Permutation.Engine.permutations(actions)

    IO.puts("Building RankTree to imitate Tournament...")
    tree = build_imitation_tree(dataset, pruned_cands, all_rankings, eval_fn, 5)

    final_error = Enum.reduce(dataset, 0, fn {_s, sl, true_rank}, acc ->
      pred_rank = TreePolicy.evaluate(tree, sl, eval_fn)
      acc + Synthex.Pure.Permutation.OracleEngine.ranking_error(pred_rank, true_rank)
    end)

    IO.puts("Final Imitation Error on Dataset: #{final_error}")
    IO.inspect(tree)
    tree
  end

  def best_leaf_ranking(dataset, all_rankings) do
    all_rankings
    |> Enum.min_by(fn ranking ->
      Enum.reduce(dataset, 0, fn {_s, _sl, true_rank}, acc ->
        acc + Synthex.Pure.Permutation.OracleEngine.ranking_error(ranking, true_rank)
      end)
    end, fn -> hd(all_rankings) end)
  end

  defp leaf_error(dataset, all_rankings) do
    best_rank = best_leaf_ranking(dataset, all_rankings)
    Enum.reduce(dataset, 0, fn {_s, _sl, true_rank}, acc ->
      acc + Synthex.Pure.Permutation.OracleEngine.ranking_error(best_rank, true_rank)
    end)
  end

  def build_imitation_tree(dataset, _cands, all_rankings, _eval_fn, 0) do
    {:leaf, best_leaf_ranking(dataset, all_rankings)}
  end

  def build_imitation_tree(dataset, cands, all_rankings, eval_fn, max_depth) do
    current_error = leaf_error(dataset, all_rankings)

    if current_error == 0 or length(dataset) == 0 do
      {:leaf, best_leaf_ranking(dataset, all_rankings)}
    else
      best_split =
        cands
        |> Task.async_stream(fn pred ->
          {true_data, false_data} = split_dataset(dataset, pred, eval_fn)
          if length(true_data) == 0 or length(false_data) == 0 do
            {pred, current_error, true_data, false_data}
          else
            err_t = leaf_error(true_data, all_rankings)
            err_f = leaf_error(false_data, all_rankings)
            {pred, err_t + err_f, true_data, false_data}
          end
        end, ordered: false)
        |> Enum.map(fn {:ok, res} -> res end)
        |> Enum.min_by(fn {_p, err, _t, _f} -> err end, fn -> nil end)

      case best_split do
        nil -> {:leaf, best_leaf_ranking(dataset, all_rankings)}
        {best_pred, split_error, true_data, false_data} ->
          if split_error >= current_error do
             {:leaf, best_leaf_ranking(dataset, all_rankings)}
          else
             true_branch = build_imitation_tree(true_data, cands, all_rankings, eval_fn, max_depth - 1)
             false_branch = build_imitation_tree(false_data, cands, all_rankings, eval_fn, max_depth - 1)
             {:branch, best_pred, true_branch, false_branch}
          end
      end
    end
  end

  defp split_dataset(dataset, pred, eval_fn) do
    Enum.split_with(dataset, fn {_s, sl, _true_rank} ->
      PredProg.eval(pred, sl, eval_fn)
    end)
  end
end
