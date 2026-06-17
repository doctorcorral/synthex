defmodule Synthex.Gym.GaProposerTest do
  use ExUnit.Case, async: true

  alias Synthex.Gym.{GaProposer, Mujoco, Oracle}

  # ── helpers: depth + well-formedness on the SERIALIZED form ─────────
  defp ser_depth("truep"), do: 0
  defp ser_depth("falsep"), do: 0
  defp ser_depth(["feat", f]) when is_list(f), do: 0
  defp ser_depth(["not", p]), do: 1 + ser_depth(p)
  defp ser_depth(["and", p, q]), do: 1 + max(ser_depth(p), ser_depth(q))
  defp ser_depth(["or", p, q]), do: 1 + max(ser_depth(p), ser_depth(q))
  defp ser_depth(other), do: flunk("malformed serialized predicate: #{inspect(other)}")

  defp ser_feats(["feat", f]), do: [f]
  defp ser_feats(["not", p]), do: ser_feats(p)
  defp ser_feats(["and", p, q]), do: ser_feats(p) ++ ser_feats(q)
  defp ser_feats(["or", p, q]), do: ser_feats(p) ++ ser_feats(q)
  defp ser_feats(_), do: []

  # Multi-class atom pool, including the composition target.
  @target ["sin_axis", 3, 0.5]
  @features [
    ["axis", 0, 0.1],
    ["axis", 5, -0.2],
    ["diag", 0, 1, 2],
    ["sq_diag", 2, 3, -1],
    ["sin_axis", 3, 0.5],
    ["cos_axis", 4, 0.0],
    ["tridiag", 0, 1, 2, 1, -1],
    ["wavelet_box", 6, -0.5, 0.5]
  ]

  # Synthetic scorer: reward = (# of target atoms in the genome) * 10
  # minus a small size penalty. The GA can only raise reward by
  # COMPOSING target atoms — exactly the composition-space search under
  # test. Asserts every candidate is well-formed and within `max_depth`
  # by recording the max depth it ever observes.
  defp build_ctx(max_depth, proposer_opts) do
    {:ok, agent} = Agent.start_link(fn -> {0, 0} end)

    stub = fn request ->
      cands = request["candidates"]

      observed =
        Enum.reduce(cands, 0, fn c, acc ->
          d = ser_depth(c)
          assert d <= max_depth, "candidate exceeds max_depth=#{max_depth}: #{inspect(c)}"
          max(acc, d)
        end)

      Agent.update(agent, fn {md, n} -> {max(md, observed), n + length(cands)} end)

      scores =
        cands
        |> Enum.with_index()
        |> Enum.map(fn {c, idx} ->
          n_target = Enum.count(ser_feats(c), &(&1 == @target))
          %{"idx" => idx, "reward" => n_target * 10.0 - length(ser_feats(c)) * 0.1, "landings" => 0}
        end)

      {:ok, %{"scores" => scores, "baseline_reward" => 0.0, "baseline_landings" => 0}}
    end

    ctx =
      Mujoco.init_context(:humanoid,
        scorer: stub,
        proposer: :ga,
        depth: max_depth,
        proposer_opts: proposer_opts,
        run_seed: 0
      )

    {ctx, agent}
  end

  defp run(ctx, bit, features \\ @features) do
    preds = List.duplicate(:truep, ctx.n_bits)
    GaProposer.optimize_bit(preds, bit, features, ctx, [0, 1])
  end

  test "evolves a composition that beats baseline, valid and within depth" do
    {ctx, agent} = build_ctx(2, %{"pop_size" => 60, "generations" => 10, "elite_frac" => 0.2})

    assert {pred, reward, baseline} = run(ctx, 0)
    assert baseline == 0.0
    assert reward > baseline
    # reaching the target at all requires at least one composed target atom
    assert reward >= 10.0

    ser = Oracle.serialize_pred(pred)
    assert ser_depth(ser) <= 2

    {max_seen, _n} = Agent.get(agent, & &1)
    assert max_seen <= 2, "scorer saw an over-depth candidate (#{max_seen})"
  end

  test "is deterministic for a fixed (run_seed, bit)" do
    {ctx, _} = build_ctx(2, %{"pop_size" => 40, "generations" => 6})
    assert run(ctx, 0) == run(ctx, 0)
  end

  test "empty feature pool yields nil" do
    {ctx, _} = build_ctx(2, %{"pop_size" => 20, "generations" => 3})
    assert run(ctx, 0, []) == nil
  end

  test "respects max_depth = 1 (atoms + single composition only)" do
    {ctx, agent} = build_ctx(1, %{"pop_size" => 40, "generations" => 8})
    result = run(ctx, 3)
    assert is_nil(result) or match?({_, _, _}, result)

    {max_seen, n} = Agent.get(agent, & &1)
    assert n > 0
    assert max_seen <= 1
  end

  test "all candidates remain well-formed under heavy mutation/crossover" do
    # High mutation + crossover stress the tree operators; the scorer's
    # ser_depth/ser_feats raise on any malformed term, so reaching the
    # end without flunk proves operator closure over valid genomes.
    {ctx, agent} =
      build_ctx(3, %{
        "pop_size" => 50,
        "generations" => 12,
        "mutation_rate" => 1.0,
        "crossover_rate" => 1.0,
        "init_compose_frac" => 0.7
      })

    _ = run(ctx, 1)
    {max_seen, n} = Agent.get(agent, & &1)
    assert n > 0
    assert max_seen <= 3
  end
end
