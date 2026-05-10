defmodule Synthex.Scoring do
  @moduledoc """
  Pluggable oracle invocation for Synthex's gym-in-the-loop synthesis.

  A **scorer** is a 1-arg function:

      (request :: map()) :: {:ok, response :: map()} | {:error, term()}

  where `request` is a JSON-serializable command for the Python oracle
  (typically `%{"cmd" => "score_bit" | "collect_states" | ..., ...}`)
  and `response` is the parsed JSON returned by the oracle.

  Synthex itself ships exactly one scorer: `Synthex.Scoring.LocalPython`,
  which forks a Python interpreter via `System.cmd` and talks to it
  through a tmpfile JSON protocol. This is the right choice when:

    * you have `gymnasium` + the relevant physics backend installed
      locally, AND
    * the search space is small enough that a single laptop can finish
      synthesis (e.g. CartPole, Pendulum, HalfCheetah-fast).

  For larger MuJoCo environments (Ant, Humanoid) or for crowd-sourced
  compute, plug in a distributed scorer — see the
  [synthex-hub](https://github.com/doctorcorral/synthex-hub) `client`
  library, which provides `Synthex.Hub.Scorer.new/1` that conforms to
  this interface and farms `score_bit` work out to a swarm of
  HTTP-pulling workers.

  ## Custom scorers

  Anything that returns the right shape works. For example, a scorer
  that records every request to disk for debugging:

      record = fn req ->
        File.write!("trace.log", Jason.encode!(req) <> "\\n", [:append])
        Synthex.Scoring.LocalPython.call(req, env_key: :ant)
      end

      Synthex.Gym.Mujoco.solve(:ant, scorer: record, ...)
  """

  @type request :: map()
  @type response :: map()
  @type t :: (request() -> {:ok, response()} | {:error, term()})

  @doc "Default scorer — runs the oracle locally via `System.cmd`."
  @spec default(atom()) :: t()
  def default(env_key), do: Synthex.Scoring.LocalPython.scorer(env_key: env_key)
end
