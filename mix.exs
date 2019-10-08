defmodule Matrex.MixProject do
  use Mix.Project

  @version "0.6.8"

  def project do
    [
      app: :matrex,
      version: @version,
      elixir: "~> 1.7",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      make_clean: ["clean"],
      make_env: %{
        "MATREX_BLAS" => Application.get_env(:matrex, :blas, System.get_env("MATREX_BLAS"))
      },
      compilers: [:elixir_make] ++ Mix.compilers(),
      aliases: aliases(),
      preferred_cli_env: ["bench.matrex": :bench, docs: :docs],
      description:
        "Blazing fast matrix library for Elixir/Erlang with native C implementation using CBLAS.",
      name: "Matrex",
      source_url: "https://github.com/versilov/matrex",
      homepage_url: "https://matrex.org",
      dialyzer: dialyzer(),
      test_coverage: [tool: ExCoveralls],
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: []
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:benchfella, "0.3.4", only: :dev},
      {:benchee, "~> 0.8", only: :bench},
      {:benchee_html, "~> 0.1", only: :bench},
      {:dialyxir, "0.5.0", only: [:dev, :test], runtime: false},
      {:mix_test_watch, "~> 0.3", only: :dev, runtime: false},
      {:excoveralls, github: "parroty/excoveralls", only: :test},
      {:ex_unit_notifier, "~> 0.1", only: :test},
      {:elixir_make, "~> 0.4", runtime: false},
      {:ex_doc, "~> 0.21", only: :dev, runtime: false},
      {:inch_ex, "~> 0.5", only: :docs},
      {:matrix, "~> 0.3.0", only: :bench},
      {:exmatrix, "~> 0.0.1", only: :bench},
      {:numexy, "~> 0.1.0", only: :bench},
      {:tensor, "~> 2.0", only: :bench}
    ] ++ maybe_stream_data()
  end

  defp maybe_stream_data() do
    if Version.match?(System.version(), "~> 1.5") do
      [{:stream_data, "~> 0.4", only: :test}]
    else
      []
    end
  end

  defp aliases() do
    [
      "bench.matrex": ["run bench/matrex.exs"]
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

  defp dialyzer() do
    [
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
  end

  defp docs() do
    [
      main: "Matrex",
      logo: "docs/matrex_logo_dark_rounded.png",
      source_ref: "v#{@version}",
      canonical: "http://hexdocs.pm/matrex",
      extras: [
        "README.md"
      ]
    ]
  end
end
