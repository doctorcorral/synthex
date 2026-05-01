defmodule Synthex.Pure.Permutation.StructuralTreeEngine do
  @moduledoc """
  A formal, rigorous synthesis engine for Permutation Trees (RankTrees).
  Aligns with the pure CSHRL methodology: it uses structural self-play rollouts 
  to evaluate tree candidates, ensuring that the synthesized policy is coinductively 
  self-consistent.

  Unlike a greedy swap sequence that gets stuck in local minima, a RankTree splits
  the state space into disjoint regions using Depth-0 predicates, and assigns an 
  independent pure Base Ranking to each region. The tree splits are chosen not by 
  classification error, but by evaluating the *actual true rollout penalty* of the 
  resulting tree across the counter-example anchors.
  """

  alias Synthex.Pure.Permutation.TreePolicy
  alias Synthex.Pure.Permutation.Oracle

  @doc """
  Computes the sum of rollout penalties for a given tree policy across a list of anchor states.
  """
  def evaluate_tree(tree, anchors, env_mod, eval_fn) do
    Enum.reduce(anchors, 0, fn state, acc ->
      # Convert tree to a mock 'policy' struct that the old oracle might expect, 
      # or write a custom tree rollout here.
      acc + penalty_rollout_tree(state, tree, env_mod, eval_fn, env_mod.oracle_horizon())
    end)
  end

  defp penalty_rollout_tree(_state, _tree, _env_mod, _eval_fn, 0), do: 0
  defp penalty_rollout_tree(state, tree, env_mod, eval_fn, k) do
    if env_mod.terminal?(state) do
      env_mod.penalty(state)
    else
      action = TreePolicy.best_action(tree, env_mod.state_to_list(state), eval_fn)
      {s_prime, accumulated_penalty} = apply_substeps(state, action, env_mod, 5, 0)
      accumulated_penalty + penalty_rollout_tree(s_prime, tree, env_mod, eval_fn, k - 1)
    end
  end

  defp apply_substeps(state, _action, _env_mod, 0, acc_penalty), do: {state, acc_penalty}
  defp apply_substeps(state, action, env_mod, steps_left, acc_penalty) do
    if env_mod.terminal?(state) do
      {state, acc_penalty}
    else
      s_prime = env_mod.step(state, action)
      apply_substeps(s_prime, action, env_mod, steps_left - 1, acc_penalty + env_mod.penalty(state))
    end
  end

  @doc """
  Greedily finds the single best Base Ranking for a specific subset of anchors.
  """
  def find_best_leaf(anchors, all_rankings, env_mod, eval_fn) do
    if length(anchors) == 0 do
      hd(all_rankings)
    else
      # Optimization: Instead of full rollouts for every leaf candidate across all states,
      # we evaluate the 1-step immediate penalty to pick a fast local leaf, 
      # OR we do full rollouts. Let's do full rollouts for purity.
      all_rankings
      |> Enum.map(fn ranking ->
        tree = {:leaf, ranking}
        score = evaluate_tree(tree, anchors, env_mod, eval_fn)
        {ranking, score}
      end)
      |> Enum.min_by(fn {_ranking, score} -> score end)
      |> elem(0)
    end
  end

  defp leaf_score(anchors, all_rankings, env_mod, eval_fn) do
    if length(anchors) == 0 do
      0
    else
      ranking = find_best_leaf(anchors, all_rankings, env_mod, eval_fn)
      evaluate_tree({:leaf, ranking}, anchors, env_mod, eval_fn)
    end
  end

  @doc """
  Greedy Decision Tree builder based on Structural Rollout Penalty.
  """
  def build_tree(anchors, _cands, all_rankings, env_mod, eval_fn, 0) do
    {:leaf, find_best_leaf(anchors, all_rankings, env_mod, eval_fn)}
  end

  def build_tree(anchors, cands, all_rankings, env_mod, eval_fn, max_depth) do
    current_score = leaf_score(anchors, all_rankings, env_mod, eval_fn)

    if current_score == 0 or length(anchors) == 0 do
      {:leaf, find_best_leaf(anchors, all_rankings, env_mod, eval_fn)}
    else
      # Find the best predicate to split the anchors
      # To avoid 1 Trillion evaluations, we parallelize this search
      best_split =
        cands
        |> Task.async_stream(fn pred ->
          {true_anchors, false_anchors} = split_anchors(anchors, pred, env_mod, eval_fn)
          
          # Only consider splits that actually divide the data
          if length(true_anchors) == 0 or length(false_anchors) == 0 do
            {pred, current_score, true_anchors, false_anchors}
          else
            # We evaluate the structural score of assigning the best leaf to each side
            score_t = leaf_score(true_anchors, all_rankings, env_mod, eval_fn)
            score_f = leaf_score(false_anchors, all_rankings, env_mod, eval_fn)
            total_split_score = score_t + score_f
            {pred, total_split_score, true_anchors, false_anchors}
          end
        end, ordered: false)
        |> Enum.map(fn {:ok, res} -> res end)
        |> Enum.min_by(fn {_p, score, _t, _f} -> score end, fn -> nil end)

      case best_split do
        nil -> {:leaf, find_best_leaf(anchors, all_rankings, env_mod, eval_fn)}
        {best_pred, split_score, true_anchors, false_anchors} ->
          # If the best split doesn't improve the structural rollout score, stop branching
          if split_score >= current_score do
             {:leaf, find_best_leaf(anchors, all_rankings, env_mod, eval_fn)}
          else
             # Recursively build sub-trees
             true_branch = build_tree(true_anchors, cands, all_rankings, env_mod, eval_fn, max_depth - 1)
             false_branch = build_tree(false_anchors, cands, all_rankings, env_mod, eval_fn, max_depth - 1)
             {:branch, best_pred, true_branch, false_branch}
          end
      end
    end
  end

  defp split_anchors(anchors, pred, env_mod, eval_fn) do
    Enum.split_with(anchors, fn s ->
      Synthex.Core.PredProg.eval(pred, env_mod.state_to_list(s), eval_fn)
    end)
  end

  @doc """
  Collects a trajectory using the tree policy and returns states that have high penalty (crashes).
  """
  def collect_cegar_states(_state, _tree, _env_mod, _eval_fn, 0), do: []
  def collect_cegar_states(state, tree, env_mod, eval_fn, n) do
    current_penalty = env_mod.penalty(state)
    is_hard = current_penalty > 1000

    if env_mod.terminal?(state) do
      if is_hard, do: [state], else: []
    else
      action = TreePolicy.best_action(tree, env_mod.state_to_list(state), eval_fn)
      {s_prime, _} = apply_substeps(state, action, env_mod, 5, 0)
      
      rest = collect_cegar_states(s_prime, tree, env_mod, eval_fn, n - 1)
      
      if is_hard do
        [state | rest]
      else
        rest
      end
    end
  end

  @doc """
  The Outer Loop: Synthesizes a Structural RankTree, tests it for counter-examples, and repeats.
  """
  def solve(env_mod, depth \\ 1, max_coeff \\ 1, max_tree_depth \\ 3, cegar_iters \\ 5) do
    IO.puts("==================================================")
    IO.puts("🌳 Initiating Pure Structural StateCEGAR (RankTree)")
    IO.puts("Environment: #{inspect(env_mod)}")
    IO.puts("Pred Depth: #{depth}, Max Tree Depth: #{max_tree_depth}, Iterations: #{cegar_iters}")
    IO.puts("==================================================\n")

    eval_fn = &Synthex.Core.ContinuousFeatures.eval_feature/2
    starts = env_mod.starts()
    actions = env_mod.actions()
    all_rankings = Synthex.Pure.Permutation.Engine.permutations(actions)

    cegar_loop(env_mod, starts, all_rankings, eval_fn, depth, max_coeff, max_tree_depth, cegar_iters, 1)
  end

  defp cegar_loop(env_mod, anchors, all_rankings, eval_fn, depth, max_coeff, max_tree_depth, max_iters, iter) do
    IO.puts("\n--- CEGAR Iteration #{iter}/#{max_iters} ---")
    IO.puts("1. Anchors: #{length(anchors)} states")

    # Generate trajectories from anchors to build features
    traj_lists = Enum.map(anchors, &env_mod.state_to_list/1)
    state_size = length(hd(traj_lists))

    features = Synthex.Core.ContinuousFeatures.generate(traj_lists, state_size, max_coeff)
    cands = Synthex.Core.CEGIS.enumerate(features, depth)
    
    # Prune predicates that don't differentiate any anchor states
    pruned_cands = Enum.filter(cands, fn pred ->
      evals = Enum.map(traj_lists, fn sl -> Synthex.Core.PredProg.eval(pred, sl, eval_fn) end)
      Enum.any?(evals, &(&1 == true)) and Enum.any?(evals, &(&1 == false))
    end)

    IO.puts("2. Predicates: #{length(pruned_cands)} active candidates (Depth 0)")
    IO.puts("3. Synthesizing Greedy Structural RankTree...")

    tree = build_tree(anchors, pruned_cands, all_rankings, env_mod, eval_fn, max_tree_depth)
    
    # Evaluate final structural score of the tree
    final_score = evaluate_tree(tree, anchors, env_mod, eval_fn)
    IO.puts("   Best Structural Rollout Score on Anchors: #{final_score}")

    if iter >= max_iters do
      IO.puts("\n==================================================")
      IO.puts("FINAL STRUCTURAL RANKTREE")
      IO.puts("==================================================")
      IO.inspect(tree)
      tree
    else
      IO.puts("4. Structural Rollout Validation (Finding CEXs)...")
      new_states = Enum.flat_map(env_mod.starts(), fn start ->
        collect_cegar_states(start, tree, env_mod, eval_fn, 150)
      end)
      
      all_new_states = new_states |> Enum.uniq()
      
      sampled_new = if length(all_new_states) > 50 do
        Enum.take_random(all_new_states, 50)
      else
        all_new_states
      end
      
      all_anchors = (anchors ++ sampled_new) |> Enum.uniq()

      if length(all_anchors) == length(anchors) do
        IO.puts("   No new counter-examples found. Pure CEGAR Converged!")
        IO.puts("\n==================================================")
        IO.puts("FINAL STRUCTURAL RANKTREE")
        IO.puts("==================================================")
        IO.inspect(tree)
        tree
      else
        IO.puts("   Discovered #{length(all_anchors) - length(anchors)} new counter-examples.")
        cegar_loop(env_mod, all_anchors, all_rankings, eval_fn, depth, max_coeff, max_tree_depth, max_iters, iter + 1)
      end
    end
  end
end
