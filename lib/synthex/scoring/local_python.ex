defmodule Synthex.Scoring.LocalPython do
  @moduledoc """
  Default scorer implementation. Runs the Python oracle for `env_key`
  in a forked process via `System.cmd`, communicating through tmpfile
  JSON. Implements the `Synthex.Scoring` shape.

  Each call:

    1. Writes `request` as JSON to a tmpfile.
    2. Spawns `python3 -u <oracle_script> req.json resp.json`,
       working directory = `:project_root` config.
    3. Reads `resp.json` back, parses it, deletes both tmpfiles.

  Configurable via `:synthex` app env:

    * `:python` — interpreter binary (default `"python3"`).
    * `:project_root` — cwd for the spawned interpreter (so the
      oracle can `import` from the synthex repo's Python tree).
      Default: three levels up from this file, which puts you at
      the synthex repo root in normal layout.
  """

  alias Synthex.Gym.Oracle

  @doc """
  Build a scorer closure that captures `env_key`. Pass to
  `Synthex.Gym.Mujoco.solve/2` (and any other entry point that
  accepts a `scorer:` opt) via `scorer: Synthex.Scoring.LocalPython.scorer(env_key: :ant)`.
  """
  @spec scorer(keyword()) :: Synthex.Scoring.t()
  def scorer(opts \\ []) do
    fn request -> call(request, opts) end
  end

  @doc """
  Direct entry point — useful for one-off testing or for custom
  scorers that wrap `LocalPython` (e.g. retry, logging, caching).

  Either pass `env_key:` in `opts` or include `"env_key"` in `request`.
  """
  @spec call(map(), keyword()) :: {:ok, map()} | {:error, term()}
  def call(request, opts \\ []) do
    env_key = Keyword.get(opts, :env_key) || Map.get(request, "env_key")

    if env_key == nil do
      {:error, "Synthex.Scoring.LocalPython: env_key not provided (pass via opts or request)"}
    else
      do_call(request, env_key)
    end
  end

  defp do_call(request, env_key) do
    script = Oracle.oracle_script(env_key)
    python = Application.get_env(:synthex, :python, "python3")
    cwd = Application.get_env(:synthex, :project_root, default_project_root())

    uid = :erlang.unique_integer([:positive])
    tmp_dir = System.tmp_dir!()
    req_file = Path.join(tmp_dir, "synthex_req_#{uid}.json")
    resp_file = Path.join(tmp_dir, "synthex_resp_#{uid}.json")

    try do
      File.write!(req_file, Jason.encode!(request))

      {output, exit_code} =
        System.cmd(python, ["-u", script, req_file, resp_file],
          stderr_to_stdout: true,
          cd: cwd
        )

      cond do
        exit_code != 0 ->
          {:error,
           "Synthex.Scoring.LocalPython: python exited #{exit_code}: #{String.slice(output, 0, 2000)}"}

        not File.exists?(resp_file) ->
          {:error, "Synthex.Scoring.LocalPython: python produced no response file"}

        true ->
          case File.read(resp_file) do
            {:ok, body} ->
              case Jason.decode(body) do
                {:ok, parsed} -> {:ok, parsed}
                {:error, reason} -> {:error, "Synthex.Scoring.LocalPython: invalid JSON: #{inspect(reason)}"}
              end

            {:error, reason} ->
              {:error, "Synthex.Scoring.LocalPython: read failed: #{inspect(reason)}"}
          end
      end
    after
      File.rm(req_file)
      File.rm(resp_file)
    end
  end

  # When this file lives at synthex/lib/synthex/scoring/local_python.ex,
  # three levels up is the repo root. For installed deps the runtime
  # path is different, so users should set `:project_root` explicitly.
  defp default_project_root do
    Path.expand("../../..", __DIR__)
  end
end
