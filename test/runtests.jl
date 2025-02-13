using HybridBlockArrays, Test, SparseArrays, LinearAlgebra

@testset "HybridBlockArrays" begin
  @testset "isempty" begin
    s = sparse(Float64.([1 0 0 0;
                         0 0 0 2;
                         3 4 0 0;
                         5 6 0 0]))
    s = Matrix(s)
    b = HybridBlockArray(s, 2; sparsitythreshold=0.5)
    for i in 3:4, j in 3:4
      @test HybridBlockArrays.tileisempty(b, i, j)
    end
    for i in 1:2, j in 3:2
      @test !HybridBlockArrays.tileisempty(b, i, j)
    end
    for i in 1:4, j in 1:2
      @test !HybridBlockArrays.tileisempty(b, i, j)
    end
    b[3, 4] += 6
    for i in 3:4, j in 3:4
      @test !HybridBlockArrays.tileisempty(b, i, j)
    end
    bcopy = Matrix(b[:, :])
    r = rand(1:10, 4, 4)
    bcopy .+= r
    b .+= r
    @test bcopy == b
  end

  @testset "indexing" begin
    A = rand(4, 4)
    b = HybridBlockArray(A, 2)
    b1212 = b[1:2, 1:2]
    b1212[1] += 30 # this is a copy
    b[1:2, 1:2] -= A[1:2, 1:2]
    for i in 1:2, j in 1:2
      @test HybridBlockArrays.tileisempty(b, i, j)
    end
  end

  @testset "transpose" begin
    for A in (rand(ComplexF64, 4, 4), sprand(ComplexF64, 4, 4, 0.9))
      A[1:2, 3:4] .= 0
      b = HybridBlockArray(A, 2)
      for i in 1:2, j in 3:4
        @test HybridBlockArrays.tileisempty(b, i, j)
      end
      for j in 1:2, i in 3:4
        @test !HybridBlockArrays.tileisempty(b, i, j)
      end
      transpose!(b)
      for i in 1:2, j in 3:4
        @test !HybridBlockArrays.tileisempty(b, i, j)
      end
      for j in 1:2, i in 3:4
        @test HybridBlockArrays.tileisempty(b, i, j)
      end
      @test transpose(A) ≈ b
      @test transpose!(b) ≈ A
    end
  end
end
