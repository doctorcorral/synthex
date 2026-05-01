defmodule Synthex.Experiment do
  @moduledoc """
  Config-driven experiment runner. Parses TOML experiment files and dispatches
  to the appropriate synthesis method.
  """

  @doc """
  Run an experiment from a TOML config file.
  """
  def run(config_path) do
    IO.puts("Loading experiment config: #{config_path}")

    config = parse_config(config_path)
    name = get_in(config, ["experiment", "name"]) || Path.basename(config_path, ".toml")
    method = get_in(config, ["experiment", "method"]) || "chain"

    IO.puts("Experiment: #{name}")
    IO.puts("Method: #{method}\n")

    result_dir = setup_results_dir(name)
    File.write!(Path.join(result_dir, "config.toml"), File.read!(config_path))
    save_git_info(result_dir)

    {chain, default} = dispatch(method, config)

    save_result(result_dir, chain, default, config)
    IO.puts("\nResults saved to: #{result_dir}")

    {chain, default}
  end

  # ── Config parsing ──────────────────────────────────────────

  defp parse_config(path) do
    case Toml.decode_file(path) do
      {:ok, config} -> config
      {:error, reason} ->
        IO.puts("Failed to parse config: #{inspect(reason)}")
        raise "Invalid TOML config: #{path}"
    end
  end

  # ── Method dispatch ─────────────────────────────────────────

  defp dispatch("chain", config) do
    {actions, default, opts} = extract_common_opts(config)
    Synthex.Gym.Chain.solve(actions, default, opts)
  end

  defp dispatch("ranking", config) do
    {actions, default, opts} = extract_common_opts(config)
    Synthex.Gym.Ranking.solve(actions, default, opts)
  end

  defp dispatch("successor", config) do
    {actions, default, opts} = extract_common_opts(config)

    succ_config = config["successor"] || %{}
    succ_opts = opts ++
      [
        lookahead: Map.get(succ_config, "lookahead", 100),
        sample_every: Map.get(succ_config, "sample_every", 10),
        succ_top_k: Map.get(succ_config, "succ_top_k", 200)
      ]

    Synthex.Gym.Successor.solve(actions, default, succ_opts)
  end

  defp dispatch("swapnet", config) do
    env_config = config["environment"] || %{}
    synth = config["synthesis"] || %{}

    opts = [
      env: String.to_atom(Map.get(env_config, "env", "lunarlander")),
      depth: Map.get(synth, "depth", 1),
      max_coeff: Map.get(synth, "max_coeff", 5),
      n_episodes: Map.get(synth, "n_episodes", 80),
      top_k: Map.get(synth, "top_k", 30),
      max_iters: Map.get(synth, "max_iters", 8),
      cegar_rounds: Map.get(synth, "cegar_rounds", 5)
    ]

    result = Synthex.Gym.SwapNetwork.solve(opts)
    {result, :swapnet}
  end

  defp dispatch("binary", config) do
    synth = config["synthesis"] || %{}
    opts = [
      depth: Map.get(synth, "depth", 1),
      max_coeff: Map.get(synth, "max_coeff", 5),
      n_episodes: Map.get(synth, "n_episodes", 30),
      top_k: Map.get(synth, "top_k", 20),
      max_iters: Map.get(synth, "max_iters", 5),
      cegar_rounds: Map.get(synth, "cegar_rounds", 3),
      max_steps: Map.get(synth, "max_steps", 2000)
    ]

    result = Synthex.Gym.Binary.solve(opts)
    {result, :binary}
  end

  defp dispatch("mujoco", config) do
    env_config = config["environment"] || %{}
    synth = config["synthesis"] || %{}
    env = String.to_atom(Map.get(env_config, "env", "inverted_pendulum"))

    opts = [
      depth: Map.get(synth, "depth", 1),
      max_coeff: Map.get(synth, "max_coeff", 5),
      n_episodes: Map.get(synth, "n_episodes", 30),
      top_k: Map.get(synth, "top_k", 20),
      max_iters: Map.get(synth, "max_iters", 5),
      cegar_rounds: Map.get(synth, "cegar_rounds", 3),
      max_steps: Map.get(synth, "max_steps", 1000),
      bits_per_dim: Map.get(synth, "bits_per_dim", 3)
    ]

    result = Synthex.Gym.Mujoco.solve(env, opts)
    {result, :mujoco}
  end

  defp dispatch("pairwise_matrix", config) do
    env_config = config["environment"] || %{}
    env_mod = resolve_pure_env(Map.get(env_config, "module", "MountainCar"))
    synth = config["synthesis"] || %{}

    depth = Map.get(synth, "depth", 1)
    max_coeff = Map.get(synth, "max_coeff", 3)
    max_fuel = Map.get(synth, "max_fuel", 100)

    result = Synthex.Pure.PairwiseMatrix.solve(env_mod, depth, max_coeff, max_fuel)
    {result, :pairwise}
  end

  defp dispatch("pure_chain", config) do
    env_config = config["environment"] || %{}
    env_mod = resolve_pure_env(Map.get(env_config, "module", "MountainCar"))
    actions = Enum.map(Map.get(env_config, "actions", []), &String.to_atom/1)
    default = String.to_atom(Map.get(env_config, "default", "do_nothing"))
    synth = config["synthesis"] || %{}

    opts = [
      depth: Map.get(synth, "depth", 0),
      max_coeff: Map.get(synth, "max_coeff", 5),
      cegar_rounds: Map.get(synth, "cegar_rounds", 3)
    ]

    Synthex.Pure.Chain.solve(env_mod, actions, default, opts)
  end

  defp dispatch(method, _config) do
    raise "Unknown synthesis method: #{method}"
  end

  # ── Shared option extraction ────────────────────────────────

  defp extract_common_opts(config) do
    env_config = config["environment"] || %{}
    synth = config["synthesis"] || %{}
    val_config = config["validation"] || %{}

    env = String.to_atom(Map.get(env_config, "env", "lunarlander"))
    actions = Enum.map(Map.get(env_config, "actions", []), &String.to_atom/1)
    default = String.to_atom(Map.get(env_config, "default", "do_nothing"))

    opts = [
      env: env,
      depth: Map.get(synth, "depth", 1),
      max_coeff: Map.get(synth, "max_coeff", 5),
      n_episodes: Map.get(synth, "n_episodes", 30),
      top_k: Map.get(synth, "top_k", 20),
      max_iters: Map.get(synth, "max_iters", 5),
      cegar_rounds: Map.get(synth, "cegar_rounds", 3),
      max_steps: Map.get(synth, "max_steps", 300),
      val_episodes: Map.get(val_config, "val_episodes", 500)
    ]

    {actions, default, opts}
  end

  # ── Pure env resolution ─────────────────────────────────────

  defp resolve_pure_env("LunarLander"), do: Synthex.Envs.LunarLander
  defp resolve_pure_env("MountainCar"), do: Synthex.Envs.MountainCar
  defp resolve_pure_env("Pendulum"), do: Synthex.Envs.Pendulum
  defp resolve_pure_env("Pong"), do: Synthex.Envs.Pong
  defp resolve_pure_env("BipedalWalker"), do: Synthex.Envs.BipedalWalker
  defp resolve_pure_env(name), do: String.to_atom("Elixir.Synthex.Envs.#{name}")

  # ── Results management ──────────────────────────────────────

  defp setup_results_dir(name) do
    base = Application.get_env(:synthex, :results_dir, "results")
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601() |> String.replace(~r/[:\.]/, "-")
    dir = Path.join([base, name, timestamp])
    File.mkdir_p!(dir)
    dir
  end

  defp save_git_info(dir) do
    case System.cmd("git", ["rev-parse", "HEAD"], stderr_to_stdout: true) do
      {sha, 0} -> File.write!(Path.join(dir, "git_sha.txt"), String.trim(sha))
      _ -> :ok
    end
  end

  defp save_result(dir, chain, default, _config) when is_list(chain) do
    policy = %{
      "chain" => Enum.map(chain, fn {pred, action} ->
        %{"pred" => inspect(pred), "action" => to_string(action)}
      end),
      "default" => to_string(default)
    }

    File.write!(Path.join(dir, "policy.json"), Jason.encode!(policy, pretty: true))
  end

  defp save_result(dir, result, _default, _config) do
    File.write!(Path.join(dir, "result.json"), Jason.encode!(%{"result" => inspect(result)}, pretty: true))
  end
end

defmodule Synthex.CLI do
  @moduledoc false

  def run do
    case System.argv() do
      [config_path] ->
        Synthex.Experiment.run(config_path)

      [] ->
        IO.puts("""
        Usage: mix synthex.run <config.toml>

        Or run directly:
          mix run -e 'Synthex.Experiment.run("experiments/tetris/successor_hybrid.toml")'
        """)

      _ ->
        IO.puts("Usage: mix synthex.run <config.toml>")
    end
  end
end
