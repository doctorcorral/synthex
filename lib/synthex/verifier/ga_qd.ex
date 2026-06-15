defmodule Synthex.Verifier.GAQD do
  @moduledoc """
  Adversarial quality-diversity verifier (experimental).

  Instead of scoring the synthesizer against random initial conditions,
  this strategy actively searches for the initial conditions where the
  current policy is most *one-bit-improvable* — the single-bit-flip
  regret, which is exactly the local move the CEGAR commit-gate makes.

  A MAP-Elites archive keeps the highest-regret seed per behavioural cell
  (binned by a low-dim fingerprint of the initial observation), so the
  output is a *diverse* spread of failure modes rather than one worst
  case. The synthesizer then:

    * builds features from trajectories rolled out of those hard seeds, and
    * scores candidates on those same hard seeds,

  concentrating refinement on genuine counterexamples.

  v1 scope: the genome is the reset seed and variation is resampling
  (integer seeds have no locality), so this is quality-diversity-guided
  adversarial *search* over seeds. It is fully compatible with the
  existing seed-based scorer — the elite seeds plug straight into
  `score_bit`. State-space perturbation genomes (requiring per-seed
  initial-state overrides in the scorer) are a documented follow-up; the
  `Synthex.Verifier` seam means that upgrade won't touch any caller.

  Robustness: if the worker does not support `eval_regret` (older image),
  the step transparently falls back to `Synthex.Verifier.RandomSeeds`.
  """

  @behaviour Synthex.Verifier

  require Logger

  alias Synthex.Gym.Mujoco
  alias Synthex.Verifier.RandomSeeds

  # Seed sampling space; avoids the validation band (10_000..10_199).
  @seed_space 9_000

  @impl true
  def supply(preds, ctx, round) do
    opts = ctx.verifier_opts || %{}
    n_out = opt(opts, "n_seeds", ctx.n_episodes)
    pool = opt(opts, "pool_size", max(n_out * 4, 32))
    generations = opt(opts, "generations", 4)
    bin_width = optf(opts, "bin_width", 0.5)

    rng = :rand.seed_s(:exsss, {round, pool, generations})

    try do
      {archive, _rng} =
        Enum.reduce(1..generations, {%{}, rng}, fn _gen, {arch, r} ->
          {seeds, r} = sample_seeds(pool, r)
          regrets = Mujoco.eval_regret(seeds, preds, ctx)
          {update_archive(arch, regrets, bin_width), r}
        end)

      elites =
        archive
        |> Map.values()
        |> Enum.sort_by(& &1["regret"], :desc)

      elite_seeds =
        elites
        |> Enum.map(& &1["seed"])
        |> Enum.take(n_out)
        |> pad_seeds(n_out)

      {states, _} = Mujoco.collect_states(preds, ctx, elite_seeds)

      Logger.info(
        "[Verifier.GAQD] round #{round}: #{map_size(archive)} QD cells, " <>
          "top regret=#{top_regret(elites)}, #{length(elite_seeds)} elite seeds"
      )

      %{
        states: states,
        seeds: elite_seeds,
        counterexamples: Enum.take(elites, n_out)
      }
    rescue
      e ->
        Logger.warning(
          "[Verifier.GAQD] regret probe failed (#{Exception.message(e)}); " <>
            "falling back to RandomSeeds for round #{round}"
        )

        RandomSeeds.supply(preds, ctx, round)
    end
  end

  defp sample_seeds(n, rng) do
    Enum.map_reduce(1..n, rng, fn _, r ->
      {u, r} = :rand.uniform_s(@seed_space, r)
      {u - 1, r}
    end)
  end

  # MAP-Elites update: keep the max-regret seed per behavioural cell.
  defp update_archive(archive, regrets, bin_width) do
    Enum.reduce(regrets, archive, fn rec, arch ->
      key = bin_key(rec["descriptor"], bin_width)
      incumbent = Map.get(arch, key)

      if incumbent == nil or rec["regret"] > incumbent["regret"] do
        Map.put(arch, key, rec)
      else
        arch
      end
    end)
  end

  defp bin_key(nil, _w), do: :nil_descriptor
  defp bin_key(descriptor, w) when is_list(descriptor) do
    Enum.map(descriptor, fn x -> floor(x / w) end)
  end

  defp pad_seeds(seeds, n) when length(seeds) >= n, do: Enum.take(seeds, n)

  defp pad_seeds(seeds, n) do
    have = MapSet.new(seeds)
    extra = Enum.reject(0..(n * 2), &MapSet.member?(have, &1)) |> Enum.take(n - length(seeds))
    seeds ++ extra
  end

  defp top_regret([]), do: "n/a"
  defp top_regret([best | _]), do: Float.round(best["regret"] * 1.0, 2)

  defp opt(opts, key, default) do
    case Map.get(opts, key, default) do
      n when is_integer(n) -> n
      n when is_float(n) -> trunc(n)
      _ -> default
    end
  end

  defp optf(opts, key, default) do
    case Map.get(opts, key, default) do
      n when is_number(n) -> n * 1.0
      _ -> default
    end
  end
end
