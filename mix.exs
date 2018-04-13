defmodule Matrex.MixProject do
  use Mix.Project

  def project do
    [
      app: :matrex,
      version: "0.1.0",
      elixir: "~> 1.6",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      dialyzer: [
        flags: [
          # --------------------------
          # Flags that DISABLE checks
          # --------------------------
          "-Wno_return",
          "-Wno_unused",
          # "-Wno_improper_lists",
          # "-Wno_fun_app",
          "-Wno_match",
          # "-Wno_opaque",
          "-Wno_fail_call",
          "-Wno_contracts",
          "-Wno_behaviours",
          # "-Wno_missing_calls",
          # "-Wno_undefined_callbacks",
          # -------------------------
          # Flags that ENABLE checks
          # ------------------------
          "-Wunmatched_returns",
          "-Werror_handling",
          "-Wrace_conditions",
          # "-Wunderspecs",
          "-Wunknown"
          # "-Woverspecs",
          # "-Wspecdiffs"
        ]
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:benchfella, "0.3.4", only: :dev},
      {:dialyxir, "0.5.0", only: [:dev, :test], runtime: false}
    ]
  end
end
