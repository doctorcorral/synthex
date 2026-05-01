import Config

config :synthex,
  python: System.get_env("SYNTHEX_PYTHON", "python3"),
  oracles_dir: Path.expand("../oracles", __DIR__),
  results_dir: Path.expand("../results", __DIR__)
