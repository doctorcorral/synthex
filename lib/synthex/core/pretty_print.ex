defmodule Synthex.Core.PrettyPrint do
  @moduledoc """
  Render bit-predicate programs as readable Python pseudocode.

  Drives the "click an experiment card to see its current policy"
  affordance on the synthex.fit landing page and any other
  human-facing surface that wants to show what the swarm has
  actually synthesized.

  The Python we emit is *executable* — given a Gym observation
  vector named `obs`, it produces the integer action (for discrete
  envs) or list of continuous actions in the configured range.

  ## Example

      iex> preds = [{:and, {:feat, ["diag", 10, 5, 2]},
      ...>                 {:feat, ["diag", 10, 15, 3]}}, :falsep, :falsep]
      iex> Synthex.Core.PrettyPrint.to_python(preds,
      ...>   bits_per_dim: 3,
      ...>   n_action_dims: 1,
      ...>   action_range: {-1.0, 1.0},
      ...>   action_dim_names: %{0 => "torque"}) |> IO.puts()
      def policy(obs):
          bits = [0] * 3
          # torque
          if (2*obs[10] + obs[5] < 0) and (3*obs[10] + obs[15] < 0): bits[0] = 1
          # bits[1], bits[2]: not yet synthesized (falsep)
      ...
  """

  alias Synthex.Core.PredProg

  @doc """
  Render a list of bit predicates to a Python function string.

  Options:

    * `:bits_per_dim`      (required)
    * `:n_action_dims`     (required)
    * `:action_range`      `{lo, hi}` tuple
    * `:action_dim_names`  map of `dim_idx => "name"`
    * `:function_name`     name of the generated function. Default
                           `"policy"`.
  """
  @spec to_python([PredProg.t()], keyword) :: String.t()
  def to_python(bit_predicates, opts) when is_list(bit_predicates) do
    bits_per_dim = Keyword.fetch!(opts, :bits_per_dim)
    n_dims = Keyword.fetch!(opts, :n_action_dims)
    {lo, hi} = Keyword.fetch!(opts, :action_range)
    dim_names = Keyword.get(opts, :action_dim_names, %{})
    fn_name = Keyword.get(opts, :function_name, "policy")

    n_bits = bits_per_dim * n_dims
    max_sum = Integer.pow(2, bits_per_dim) - 1
    span = hi - lo

    bits_section =
      bit_predicates
      |> Enum.with_index()
      |> Enum.chunk_every(bits_per_dim)
      |> Enum.with_index()
      |> Enum.map_join("\n", fn {dim_chunk, dim_idx} ->
        dim_name = Map.get(dim_names, dim_idx, "dim#{dim_idx}")
        header = "    # #{dim_name}"

        lines =
          dim_chunk
          |> Enum.map_join("\n", fn {pred, bit_idx} -> render_bit_line(pred, bit_idx) end)

        "#{header}\n#{lines}"
      end)

    weights_list =
      for i <- 0..(bits_per_dim - 1) do
        Integer.pow(2, i) |> Integer.to_string()
      end
      |> Enum.join(", ")

    actions_block =
      [
        "    weights = [#{weights_list}]",
        "    actions = [0.0] * #{n_dims}",
        "    for d in range(#{n_dims}):",
        "        s = sum(weights[i] * bits[d*#{bits_per_dim}+i] for i in range(#{bits_per_dim}))",
        "        actions[d] = #{render_float(lo)} + #{render_float(span)} * s / #{max_sum}",
        "    return actions"
      ]
      |> Enum.join("\n")

    """
    def #{fn_name}(obs):
        bits = [0] * #{n_bits}
    #{bits_section}

    #{actions_block}
    """
    |> String.trim_trailing()
    |> Kernel.<>("\n")
  end

  defp render_bit_line(:falsep, bit_idx) do
    # falsep means "this bit is permanently 0"; we still emit the
    # line as a no-op comment for transparency rather than silently
    # skipping it.
    "    # bits[#{bit_idx}]: falsep (always 0, not yet synthesized)"
  end

  defp render_bit_line(:truep, bit_idx) do
    "    bits[#{bit_idx}] = 1  # truep (always 1)"
  end

  defp render_bit_line(pred, bit_idx) do
    "    if #{render_pred(pred)}: bits[#{bit_idx}] = 1"
  end

  @doc """
  Render a single `PredProg.t()` as a parenthesized Python boolean
  expression. Used by `to_python/2` but also handy for tests and
  ad-hoc inspection.
  """
  def render_pred(:truep), do: "True"
  def render_pred(:falsep), do: "False"
  def render_pred({:feat, spec}), do: render_feature(spec)
  def render_pred({:not, p}), do: "not (#{render_pred(p)})"
  def render_pred({:and, a, b}), do: "(#{render_pred(a)}) and (#{render_pred(b)})"
  def render_pred({:or, a, b}), do: "(#{render_pred(a)}) or (#{render_pred(b)})"

  # ── Feature rendering ────────────────────────────────────────
  # Mirrors the cases in `Synthex.Gym.Oracle.generate_features/2`.
  # Keep these in sync if/when new feature classes land.

  defp render_feature(["axis", i, t]) do
    "obs[#{i}] < #{render_float(t)}"
  end

  defp render_feature(["diag", i, j, c]) do
    "#{render_coeff(c)}*obs[#{i}] + obs[#{j}] < 0"
  end

  defp render_feature(["sq_diag", i, j, c]) do
    "#{render_coeff(c)}*obs[#{i}]**2 + obs[#{j}] < 0"
  end

  defp render_feature(["prod", i, j, t]) do
    "obs[#{i}] * obs[#{j}] < #{render_float(t)}"
  end

  defp render_feature(["tridiag", i, j, k, c1, c2]) do
    "#{render_coeff(c1)}*obs[#{i}] + #{render_coeff(c2)}*obs[#{j}] + obs[#{k}] < 0"
  end

  defp render_feature(other) do
    # Unknown feature spec — surface it as an opaque comment-marker
    # rather than crashing the whole render.
    "/* unknown feature #{inspect(other)} */"
  end

  defp render_coeff(c) when is_integer(c), do: Integer.to_string(c)

  defp render_coeff(c) when is_float(c) do
    # Whole-number floats render as ints for readability.
    if c == Float.round(c) and abs(c) < 1.0e15 do
      Integer.to_string(trunc(c))
    else
      render_float(c)
    end
  end

  defp render_float(n) when is_integer(n), do: "#{n}.0"

  defp render_float(n) when is_float(n) do
    cond do
      n == Float.round(n) and abs(n) < 1.0e15 ->
        "#{trunc(n)}.0"

      true ->
        # 6 sig figs is plenty for inspecting policies; the actual
        # synthesis uses full precision internally.
        :erlang.float_to_binary(n, decimals: 6)
        |> String.trim_trailing("0")
        |> String.trim_trailing(".")
    end
  end

  @doc """
  Convert a `PredProg.t()` to a JSON-safe term — tuples become
  tagged lists. Used by masters when shipping snapshots to the
  hub over HTTP.

      :truep                       -> "truep"
      :falsep                      -> "falsep"
      {:feat, X}                   -> ["feat", X]
      {:not, P}                    -> ["not", encode(P)]
      {:and, A, B}                 -> ["and", encode(A), encode(B)]
      {:or, A, B}                  -> ["or", encode(A), encode(B)]
  """
  def to_json_term(:truep), do: "truep"
  def to_json_term(:falsep), do: "falsep"
  def to_json_term({:feat, spec}), do: ["feat", spec]
  def to_json_term({:not, p}), do: ["not", to_json_term(p)]
  def to_json_term({:and, a, b}), do: ["and", to_json_term(a), to_json_term(b)]
  def to_json_term({:or, a, b}), do: ["or", to_json_term(a), to_json_term(b)]
end
