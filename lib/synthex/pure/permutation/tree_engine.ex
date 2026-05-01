defmodule Synthex.Pure.Permutation.TreeEngine do
  @moduledoc """
  Synthesizes a Permutation Tree (RankTree) using greedy Information Gain / Error Reduction.
  This bypasses the combinatorial explosion of beam-searching flat lists, 
  allowing ultra-fast fitting to the Oracle Dataset using only Depth-0 predicates.
  """

  alias Synthex.Pure.Permutation.TreePolicy

  @doc """
  Finds the single static ranking that minimizes error for a given dataset.
  Used to create Leaf nodes.
  """
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

  @doc """
  Greedy Decision Tree builder.
  """
  def build_tree(dataset, _cands, all_rankings, _eval_fn, 0) do
    {:leaf, best_leaf_ranking(dataset, all_rankings)}
  end

  def build_tree(dataset, cands, all_rankings, eval_fn, max_depth) do
    current_error = leaf_error(dataset, all_rankings)

    # If the dataset is perfectly classified or empty, return a leaf
    if current_error == 0 or length(dataset) == 0 do
      {:leaf, best_leaf_ranking(dataset, all_rankings)}
    else
      # Find the best predicate to split the data
      best_split =
        cands
        |> Task.async_stream(fn pred ->
          {true_data, false_data} = split_dataset(dataset, pred, eval_fn)
          
          # We only consider splits that actually divide the data
          if length(true_data) == 0 or length(false_data) == 0 do
            {pred, current_error, true_data, false_data}
          else
            err_t = leaf_error(true_data, all_rankings)
            err_f = leaf_error(false_data, all_rankings)
            total_split_error = err_t + err_f
            {pred, total_split_error, true_data, false_data}
          end
        end, ordered: false)
        |> Enum.map(fn {:ok, res} -> res end)
        |> Enum.min_by(fn {_p, err, _t, _f} -> err end, fn -> nil end)

      case best_split do
        nil -> {:leaf, best_leaf_ranking(dataset, all_rankings)}
        {best_pred, split_error, true_data, false_data} ->
          # If the best split doesn't improve the error at all, stop branching to prevent infinite recursion on noisy data
          if split_error >= current_error do
             {:leaf, best_leaf_ranking(dataset, all_rankings)}
          else
             true_branch = build_tree(true_data, cands, all_rankings, eval_fn, max_depth - 1)
             false_branch = build_tree(false_data, cands, all_rankings, eval_fn, max_depth - 1)
             {:branch, best_pred, true_branch, false_branch}
          end
      end
    end
  end

  defp split_dataset(dataset, pred, eval_fn) do
    Enum.split_with(dataset, fn {_s, sl, _true_rank} ->
      Synthex.Core.PredProg.eval(pred, sl, eval_fn)
    end)
  end

  @doc """
  Main synthesis loop with CEGAR dataset expansion using Tree building.
  """
  def solve(env_mod, max_coeff \\ 1, max_tree_depth \\ 5, cegar_iters \\ 8) do
    IO.puts("==================================================")
    IO.puts("🌲 Initiating StateCEGAR Oracle-Guided RankTree")
    IO.puts("Environment: #{inspect(env_mod)}")
    IO.puts("Max Tree Depth: #{max_tree_depth}, Iterations: #{cegar_iters}")
    IO.puts("==================================================\n")

    eval_fn = &Synthex.Core.ContinuousFeatures.eval_feature/2
    starts = env_mod.starts()
    
    # We expand the dataset slightly by stepping the environment
    initial_states = Enum.flat_map(starts, fn s -> 
      [s, env_mod.step(s, hd(env_mod.actions()))] 
    end) |> Enum.uniq()

    cegar_loop(env_mod, initial_states, eval_fn, max_coeff, max_tree_depth, cegar_iters, 1)
  end

  defp cegar_loop(env_mod, states, eval_fn, max_coeff, max_tree_depth, max_iters, iter) do
    IO.puts("\n--- CEGAR Iteration #{iter}/#{max_iters} ---")
    IO.puts("1. Oracle Labeling #{length(states)} states...")
    
    dataset = Enum.map(states, fn s ->
      true_rank = Synthex.Pure.Permutation.OracleEngine.get_true_ranking(s, env_mod)
      {s, env_mod.state_to_list(s), true_rank}
    end)

    IO.puts("2. Generating and Pruning Predicates...")
    traj_lists = Enum.map(dataset, fn {_, sl, _} -> sl end)
    state_size = length(hd(traj_lists))

    # We only use Depth 0 for trees, the tree itself provides the logical depth!
    features = Synthex.Core.ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
    cands = Synthex.Core.CEGIS.enumerate(features, 0)
    
    pruned_cands = Enum.filter(cands, fn pred ->
      evals = Enum.map(dataset, fn {_, sl, _} -> Synthex.Core.PredProg.eval(pred, sl, eval_fn) end)
      Enum.any?(evals, &(&1 == true)) and Enum.any?(evals, &(&1 == false))
    end)
    
    IO.puts("   Total Depth-0 predicates: #{length(cands)}")
    IO.puts("   Pruned active predicates: #{length(pruned_cands)}")

    IO.puts("3. Synthesizing RankTree...")
    actions = env_mod.actions()
    all_rankings = Synthex.Pure.Permutation.Engine.permutations(actions)

    tree = build_tree(dataset, pruned_cands, all_rankings, eval_fn, max_tree_depth)
    
    # Evaluate final error
    final_error = Enum.reduce(dataset, 0, fn {_s, sl, true_rank}, acc ->
      pred_rank = TreePolicy.evaluate(tree, sl, eval_fn)
      acc + Synthex.Pure.Permutation.OracleEngine.ranking_error(pred_rank, true_rank)
    end)
    
    IO.puts("   Best Classification Error on Dataset: #{final_error}")

    if iter >= max_iters do
      IO.puts("\n==================================================")
      IO.puts("🏆 FINAL ORACLE-GUIDED RANKTREE")
      IO.puts("==================================================")
      IO.inspect(tree)
      tree
    else
      IO.puts("4. CEGAR Rollout Validation...")
      # Collect new states using the synthesized tree
      new_states = Enum.flat_map(env_mod.starts(), fn start ->
        collect_cegar_states(start, tree, env_mod, eval_fn, 50)
      end)
      
      all_new_states = new_states |> Enum.uniq()
      
      sampled_new = if length(all_new_states) > 100 do
        Enum.take_random(all_new_states, 100)
      else
        all_new_states
      end
      
      all_states = (states ++ sampled_new) |> Enum.uniq()
      
      if length(all_states) == length(states) do
        IO.puts("   No new states discovered. CEGAR Converged!")
        IO.puts("\n==================================================")
        IO.puts("🏆 FINAL ORACLE-GUIDED RANKTREE")
        IO.puts("==================================================")
        IO.inspect(tree)
        tree
      else
        IO.puts("   Discovered #{length(all_states) - length(states)} new counter-example states.")
        cegar_loop(env_mod, all_states, eval_fn, max_coeff, max_tree_depth, max_iters, iter + 1)
      end
    end
  end

  defp collect_cegar_states(_state, _tree, _env_mod, _eval_fn, 0), do: []
  defp collect_cegar_states(state, tree, env_mod, eval_fn, n) do
    current_penalty = env_mod.penalty(state)
    is_hard = current_penalty > 1000

    if env_mod.terminal?(state) do
      if is_hard, do: [state], else: []
    else
      ranked_list = TreePolicy.evaluate(tree, env_mod.state_to_list(state), eval_fn)
      action = hd(ranked_list)
      s_prime = env_mod.step(state, action)
      
      rest = collect_cegar_states(s_prime, tree, env_mod, eval_fn, n - 1)
      
      if is_hard do
        [state | rest]
      else
        rest
      end
    end
  end
end
