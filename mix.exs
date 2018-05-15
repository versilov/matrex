defmodule Matrex.MixProject do
  use Mix.Project

  def project do
    [
      app: :matrex,
      version: "0.4.2",
      elixir: "~> 1.4",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      compilers: [:elixir_make] ++ Mix.compilers(),
      description:
        "Fast matrix manipulation library for Elixir with native C implementation using CBLAS.",
      source_url: "https://github.com/versilov/matrex",
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
      ],
      test_coverage: [tool: ExCoveralls]
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
      {:dialyxir, "0.5.0", only: [:dev, :test], runtime: false},
      {:mix_test_watch, "~> 0.3", only: :dev, runtime: false},
      {:excoveralls, github: "parroty/excoveralls", only: :test},
      {:elixir_make, "~> 0.4", runtime: false},
      {:ex_doc, ">= 0.0.0", only: :dev},
      {:inch_ex, "~> 0.5", only: :docs}
    ]
  end

  defp package do
    [
      files: [
        "lib",
        "native",
        "Makefile",
        "README.md",
        "mix.exs"
      ],
      maintainers: [
        "Stas Versilov"
      ],
      licenses: ["simplified BSD"],
      build_tools: ["make"],
      links: %{
        "GitHub" => "https://github.com/versilov/matrex"
      }
    ]
  end
end
