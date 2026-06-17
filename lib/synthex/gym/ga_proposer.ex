defmodule Synthex.Gym.GaProposer do
  @moduledoc """
  Genetic per-bit candidate proposer (pluggable via `proposer: :ga`).

  The default `:enumerate` proposer materializes the depth-0 atom pool,
  sub-samples it to `max_candidates`, and grows depth by composing the
  `top_k` best atoms of the previous level. That top_k deepening only
  ever explores `top_k²` compositions per level — fine when the atom
  pool is small, but a vanishing slice of the space once the pool itself
  is a sample of millions (Humanoid: 105 obs dims). The composition
  space (`pool²` at depth 1, `pool³` at depth 2) is the genuinely
  non-enumerable part, and that is exactly where a fitness-guided search
  earns its keep.

  This proposer runs a genetic algorithm over **predicate programs**
  built from the supplied atom pool:

    * genome = a predicate tree — an atom `{:feat, f}` or an AND/OR/NOT
      composition of atoms, bounded by `max_depth`;
    * atoms are drawn from the `features` pool, which already spans every
      enabled feature class (axis, diag, sq_diag, tridiag, sin/cos,
      prod, wavelets — controlled by `feature_types`), so the genome can
      freely mix classes;
    * fitness = the bit's scored return (the same `score_bit` batch the
      enumerate path uses), so generations dispatch to the worker swarm
      exactly like normal scoring;
    * variation = subtree crossover + mutation (point-replace an atom,
      wrap an atom into a composition, flip AND↔OR, replace whole).

  Atoms stay valid-by-construction (every atom is a real pool feature),
  so no malformed predicate can reach the scorer. The search novelty is
  entirely in *how atoms are composed*. Numeric field mutation (jittering
  a coefficient/threshold off the pool grid) is a documented follow-up.

  Returns the same contract as the enumerate path: `{pred, reward,
  baseline}` when the best genome beats the current predicate vector on
  the scoring seeds, otherwise `nil`.

  ## Options (`proposer_opts`)

    * `pop_size`        (256)  — genomes per generation
    * `generations`     (8)    — number of generations
    * `elite_frac`      (0.15) — top fraction carried over unchanged
    * `tournament_k`    (3)    — tournament size for parent selection
    * `crossover_rate`  (0.6)  — probability a child is a crossover
    * `mutation_rate`   (0.8)  — probability a child is then mutated
    * `max_depth`       (max(ctx.depth, 1)) — composition depth cap
    * `init_compose_frac` (0.4) — fraction of the initial population that
      starts as depth-1 compositions (rest are bare atoms)
  """

  require Logger

  alias Synthex.Gym.Mujoco

  @spec optimize_bit(
          [Synthex.Core.PredProg.predicate()],
          non_neg_integer(),
          [term()],
          map(),
          [non_neg_integer()]
        ) :: {Synthex.Core.PredProg.predicate(), float(), float()} | nil
  def optimize_bit(_preds, _bit_idx, [], _ctx, _seeds), do: nil

  def optimize_bit(preds, bit_idx, features, ctx, seeds) do
    opts = ctx[:proposer_opts] || %{}
    pop_size = opt_int(opts, "pop_size", 256)
    generations = opt_int(opts, "generations", 8)
    tournament_k = opt_int(opts, "tournament_k", 3)
    elite_frac = opt_float(opts, "elite_frac", 0.15) |> clamp(0.0, 0.9)
    crossover_rate = opt_float(opts, "crossover_rate", 0.6) |> clamp(0.0, 1.0)
    mutation_rate = opt_float(opts, "mutation_rate", 0.8) |> clamp(0.0, 1.0)
    max_depth = opt_int(opts, "max_depth", max(Map.get(ctx, :depth, 1), 1))
    init_compose_frac = opt_float(opts, "init_compose_frac", 0.4) |> clamp(0.0, 1.0)

    atoms = List.to_tuple(Enum.map(features, fn f -> {:feat, f} end))
    n_atoms = tuple_size(atoms)
    num_dims = ctx.cfg.num_dims

    cfg = %{
      atoms: atoms,
      n_atoms: n_atoms,
      num_dims: num_dims,
      max_depth: max_depth,
      tournament_k: tournament_k,
      crossover_rate: crossover_rate,
      mutation_rate: mutation_rate,
      elite_count: max(trunc(pop_size * elite_frac), 1)
    }

    # Deterministic per (run_seed, bit) so a resumed bit re-explores the
    # same search; :rand.seed_s/2 wants a 3-int tuple.
    run_seed = Map.get(ctx, :run_seed, 0)
    rng = :rand.seed_s(:exsss, {run_seed + 1, bit_idx + 1, pop_size + generations})

    {pop, rng} = init_population(pop_size, init_compose_frac, cfg, rng)

    {best_pred, best_reward, baseline, _pop, _rng} =
      Enum.reduce(1..generations, {nil, nil, nil, pop, rng}, fn _gen,
                                                                {bp, br, base, population, r} ->
        {fitness, gen_baseline} = score_population(population, preds, bit_idx, seeds, ctx)
        base = base || gen_baseline

        {gbp, gbr} = best_of_generation(population, fitness)
        {bp, br} = keep_best({bp, br}, {gbp, gbr})

        {next_pop, r} = evolve(population, fitness, cfg, r)
        {bp, br, base, next_pop, r}
      end)

    baseline = baseline || 0.0

    cond do
      is_nil(best_pred) or not is_number(best_reward) -> nil
      best_reward > baseline -> {best_pred, best_reward, baseline}
      true -> nil
    end
  end

  # ── Population init ────────────────────────────────────────────────

  defp init_population(pop_size, compose_frac, cfg, rng) do
    Enum.map_reduce(1..pop_size, rng, fn _i, r ->
      {u, r} = rand_float(r)

      if u < compose_frac and cfg.max_depth >= 1 do
        random_composition(cfg, r)
      else
        random_atom(cfg, r)
      end
    end)
  end

  defp random_atom(cfg, rng) do
    {k, rng} = rand_int(cfg.n_atoms, rng)
    {elem(cfg.atoms, k), rng}
  end

  defp random_composition(cfg, rng) do
    {a, rng} = random_atom(cfg, rng)
    {b, rng} = random_atom(cfg, rng)
    {op, rng} = random_op(rng)

    case op do
      :not -> {{:not, a}, rng}
      _ -> {{op, a, b}, rng}
    end
  end

  defp random_op(rng) do
    {u, rng} = rand_float(rng)

    cond do
      u < 0.45 -> {:and, rng}
      u < 0.9 -> {:or, rng}
      true -> {:not, rng}
    end
  end

  # ── Fitness ────────────────────────────────────────────────────────

  defp score_population(population, preds, bit_idx, seeds, ctx) do
    unique = Enum.uniq(population)
    {scored, baseline} = Mujoco.score_candidates(unique, preds, bit_idx, seeds, ctx)

    by_idx = Map.new(scored, fn {idx, reward, _l} -> {idx, reward} end)

    fitness =
      unique
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {genome, idx}, acc ->
        case Map.get(by_idx, idx) do
          r when is_number(r) -> Map.put(acc, genome, r)
          _ -> acc
        end
      end)

    {fitness, baseline}
  end

  defp best_of_generation(population, fitness) do
    population
    |> Enum.uniq()
    |> Enum.reduce({nil, nil}, fn genome, {bp, br} ->
      case Map.get(fitness, genome) do
        r when is_number(r) and (br == nil or r > br) -> {genome, r}
        _ -> {bp, br}
      end
    end)
  end

  defp keep_best({bp, br}, {gbp, gbr}) do
    cond do
      not is_number(gbr) -> {bp, br}
      br == nil or gbr > br -> {gbp, gbr}
      true -> {bp, br}
    end
  end

  # ── Selection + reproduction ───────────────────────────────────────

  defp evolve(population, fitness, cfg, rng) do
    ranked =
      population
      |> Enum.uniq()
      |> Enum.filter(&Map.has_key?(fitness, &1))
      |> Enum.sort_by(fn g -> -Map.fetch!(fitness, g) end)

    # No genome scored (all NaN): reseed from atoms so the search doesn't
    # collapse to an empty pool.
    if ranked == [] do
      init_population(length(population), 0.3, cfg, rng)
    else
      elites = Enum.take(ranked, min(cfg.elite_count, length(ranked)))
      ranked_t = List.to_tuple(ranked)
      n_children = max(length(population) - length(elites), 0)

      {children, rng} =
        Enum.map_reduce(1..max(n_children, 1), rng, fn _i, r ->
          breed_child(ranked_t, fitness, cfg, r)
        end)

      {elites ++ Enum.take(children, n_children), rng}
    end
  end

  defp breed_child(ranked_t, fitness, cfg, rng) do
    {p1, rng} = tournament(ranked_t, fitness, cfg.tournament_k, rng)
    {u_x, rng} = rand_float(rng)

    {child, rng} =
      if u_x < cfg.crossover_rate do
        {p2, rng} = tournament(ranked_t, fitness, cfg.tournament_k, rng)
        crossover(p1, p2, cfg, rng)
      else
        {p1, rng}
      end

    {u_m, rng} = rand_float(rng)

    if u_m < cfg.mutation_rate do
      mutate(child, cfg, rng)
    else
      {child, rng}
    end
  end

  defp tournament(ranked_t, fitness, k, rng) do
    n = tuple_size(ranked_t)

    {best, rng} =
      Enum.reduce(1..max(k, 1), {nil, rng}, fn _i, {acc, r} ->
        {idx, r} = rand_int(n, r)
        cand = elem(ranked_t, idx)

        case acc do
          nil -> {cand, r}
          _ -> if Map.fetch!(fitness, cand) > Map.fetch!(fitness, acc), do: {cand, r}, else: {acc, r}
        end
      end)

    {best, rng}
  end

  # ── Genetic operators (closed over valid genomes) ──────────────────

  defp crossover(g1, g2, cfg, rng) do
    {donor, rng} = random_subtree(g2, rng)
    {child, rng} = replace_random_subtree(g1, donor, rng)

    if depth(child) <= cfg.max_depth do
      {child, rng}
    else
      {g1, rng}
    end
  end

  defp mutate(g, cfg, rng) do
    {u, rng} = rand_float(rng)

    cond do
      u < 0.4 ->
        # point-replace a random subtree with a random atom
        {atom, rng} = random_atom(cfg, rng)
        replace_random_subtree(g, atom, rng)

      u < 0.7 ->
        # wrap into a composition with a random atom (respect depth cap)
        {atom, rng} = random_atom(cfg, rng)
        {op, rng} = random_op(rng)

        wrapped =
          case op do
            :not -> {:not, g}
            _ -> {op, g, atom}
          end

        if depth(wrapped) <= cfg.max_depth do
          {wrapped, rng}
        else
          {atom2, rng} = random_atom(cfg, rng)
          replace_random_subtree(g, atom2, rng)
        end

      u < 0.85 ->
        flip_random_op(g, rng)

      true ->
        # replace whole genome
        random_atom(cfg, rng)
    end
  end

  # ── Tree utilities ─────────────────────────────────────────────────

  defp depth({:feat, _}), do: 0
  defp depth({:not, p}), do: 1 + depth(p)
  defp depth({op, a, b}) when op in [:and, :or], do: 1 + max(depth(a), depth(b))

  defp subtrees({:feat, _} = g), do: [g]
  defp subtrees({:not, p} = g), do: [g | subtrees(p)]
  defp subtrees({op, a, b} = g) when op in [:and, :or], do: [g | subtrees(a) ++ subtrees(b)]

  defp random_subtree(g, rng) do
    subs = subtrees(g)
    {k, rng} = rand_int(length(subs), rng)
    {Enum.at(subs, k), rng}
  end

  defp count_nodes({:feat, _}), do: 1
  defp count_nodes({:not, p}), do: 1 + count_nodes(p)
  defp count_nodes({op, a, b}) when op in [:and, :or], do: 1 + count_nodes(a) + count_nodes(b)

  defp replace_random_subtree(g, repl, rng) do
    {k, rng} = rand_int(count_nodes(g), rng)
    {new_g, _} = do_replace(g, repl, k)
    {new_g, rng}
  end

  # Preorder replace of the `idx`-th node. Threads remaining index; the
  # `:done` sentinel short-circuits once the replacement is placed.
  defp do_replace(_g, repl, 0), do: {repl, :done}

  defp do_replace({:feat, _} = g, _repl, idx), do: {g, idx - 1}

  defp do_replace({:not, p}, repl, idx) do
    {p2, rest} = do_replace(p, repl, idx - 1)
    {{:not, p2}, rest}
  end

  defp do_replace({op, a, b}, repl, idx) when op in [:and, :or] do
    {a2, rest} = do_replace(a, repl, idx - 1)

    case rest do
      :done ->
        {{op, a2, b}, :done}

      n ->
        {b2, rest2} = do_replace(b, repl, n)
        {{op, a2, b2}, rest2}
    end
  end

  defp flip_random_op(g, rng) do
    ops = collect_op_paths(g, [])

    case ops do
      [] ->
        {g, rng}

      _ ->
        {k, rng} = rand_int(length(ops), rng)
        path = Enum.at(ops, k)
        {flip_op_at(g, path), rng}
    end
  end

  # Paths (lists of :a/:b/:p steps) to every AND/OR node.
  defp collect_op_paths({:feat, _}, _path), do: []
  defp collect_op_paths({:not, p}, path), do: collect_op_paths(p, [:p | path])

  defp collect_op_paths({op, a, b}, path) when op in [:and, :or] do
    [Enum.reverse(path)] ++
      collect_op_paths(a, [:a | path]) ++
      collect_op_paths(b, [:b | path])
  end

  defp flip_op_at({op, a, b}, []) when op in [:and, :or] do
    {if(op == :and, do: :or, else: :and), a, b}
  end

  defp flip_op_at({:not, p}, [:p | rest]), do: {:not, flip_op_at(p, rest)}
  defp flip_op_at({op, a, b}, [:a | rest]) when op in [:and, :or], do: {op, flip_op_at(a, rest), b}
  defp flip_op_at({op, a, b}, [:b | rest]) when op in [:and, :or], do: {op, a, flip_op_at(b, rest)}

  # ── RNG + opt helpers ──────────────────────────────────────────────

  defp rand_float(rng), do: :rand.uniform_s(rng)

  defp rand_int(n, rng) when n <= 1, do: {0, rng}

  defp rand_int(n, rng) do
    {i, rng} = :rand.uniform_s(n, rng)
    {i - 1, rng}
  end

  defp clamp(x, lo, hi), do: x |> max(lo) |> min(hi)

  defp opt_int(opts, key, default) do
    case Map.get(opts, key, default) do
      n when is_integer(n) -> n
      n when is_float(n) -> trunc(n)
      _ -> default
    end
  end

  defp opt_float(opts, key, default) do
    case Map.get(opts, key, default) do
      n when is_number(n) -> n * 1.0
      _ -> default
    end
  end
end
