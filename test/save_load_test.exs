defmodule SaveLoadTest do
  use ExUnit.Case, async: true
  import Matrex

  test "#load loads matrex from CSV" do
    m = Matrex.load("test/data/matrex.csv")

    expected =
      Matrex.new([
        [0, 0.0004825367647058945, -0.005166973039215696, -0.01552127919774978],
        [-0.01616482843137255, -0.01621783088235294, -0.01609620098039215, -0.005737132352941162],
        [0.0006824018588724531, 0, 0, 0],
        [0, 0, 0, 0],
        [1.153158165611269e-31, 3.785249770368984e-18, 0, 0]
      ])

    assert m == expected
  end

  test "#load raises, when format is unkonwn" do
    assert_raise RuntimeError, fn ->
      Matrex.load("test/data/matrex.txt")
    end
  end

  test "#save raises, when file format is unkonwn" do
    assert_raise RuntimeError, fn ->
      Matrex.random(3) |> Matrex.save("m.txt")
    end
  end

  @test_file_name_idx "_test_m.idx"
  @test_file_name_csv "_test_m.csv"
  @test_file_name_idx_gzip "_test_m.idx.gz"
  @test_file_name_csv_gzip "_test_m.csv.gz"

  test "Saves to and loads from binary .idx format" do
    m = Matrex.random(100)
    Matrex.save(m, @test_file_name_idx)
    l = Matrex.load(@test_file_name_idx)
    assert l == m
    assert File.rm(@test_file_name_idx) == :ok
  end

  test "Saves to and loads from binary .idx format matrix with NaNs and Infs " do
    m = Matrex.divide(Matrex.eye(50), Matrex.zeros(50))
    Matrex.save(m, @test_file_name_idx)
    l = Matrex.load(@test_file_name_idx)
    assert l == m
    assert File.rm(@test_file_name_idx) == :ok
  end

  test "Saves to and loads from .csv format" do
    m = Matrex.random(100)
    Matrex.save(m, @test_file_name_csv)
    l = Matrex.load(@test_file_name_csv)
    assert l == m
    assert File.rm(@test_file_name_csv) == :ok
  end

  test "Saves to and loads from CSV matirx with special values." do
    m = Matrex.divide(Matrex.eye(50), Matrex.zeros(50))
    Matrex.save(m, @test_file_name_csv)
    l = Matrex.load(@test_file_name_csv)
    assert l == m
    assert File.rm(@test_file_name_csv) == :ok
  end

  test "#load loads matrix from .idx format" do
    m = Matrex.load("test/data/t10k-labels-idx1-ubyte.idx")
    assert Matrex.shape(m) == {10_000}
  end

  test "#load loads matrix from .idx format with explicit format set" do
    m = Matrex.load("test/data/t10k-labels-idx1-ubyte.idx", :idx)
    assert Matrex.shape(m) == {10_000}
  end

  test "#load loads matrix from .idx.gz format" do
    m = Matrex.load("test/data/t10k-images-idx3-ubyte.idx.gz")
    assert Matrex.shape(m) == {10_000, 28, 28}
  end

  test "#save saves to gziped idx format with explicit format set" do
    m = random({100, 80})
    save(m, @test_file_name_idx_gzip, format: :idx, gzip: true)

    l = load(@test_file_name_idx_gzip)
    assert l == m
    assert File.rm(@test_file_name_idx_gzip) == :ok
  end
end
