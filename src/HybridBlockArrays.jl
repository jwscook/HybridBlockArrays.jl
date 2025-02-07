module HybridBlockArrays

using ChunkSplitters, LinearAlgebra, SparseArrays

sparsity(A::SparseMatrixCSC) = nnz(A) / length(A)
sparsity(A) = count(!iszero, A) / length(A)

struct HybridBlockArray{T} <: AbstractArray{T, 2}
  denses::Matrix{Matrix{T}}
  sparses::Matrix{SparseMatrixCSC{T}}
  rowindices::Vector{UnitRange{Int64}}
  colindices::Vector{UnitRange{Int64}}
  isempties::Matrix{Bool}
  isdenses::Matrix{Bool}
  globalsize::Tuple{Int,Int}
  istransposed::Ref{Bool}
  function HybridBlockArray(A::AbstractMatrix{T}, rowindices::Vector{UnitRange{Int}}, colindices::Vector{UnitRange{Int}};
      sparsitythreshold = 0.2) where T
    denses = Matrix{Matrix{T}}(undef, length(rowindices), length(colindices))
    sparses = Matrix{SparseMatrixCSC{eltype(A)}}(undef, length(rowindices), length(colindices))
    isempties = zeros(Bool, length(rowindices), length(colindices))
    isdenses = Matrix{Bool}(undef, length(rowindices), length(colindices))
    for (i, r) in enumerate(rowindices), (j, c) in enumerate(colindices)
      tile = view(A, r, c)
      sp = sparsity(tile)
      if iszero(sp)
        isempties[i, j] = true
        denses[i, j] = zeros(T, size(tile)...)
        sparses[i, j] = spzeros(T, size(tile)...)
        isdenses[i, j] = false
      elseif sp > sparsitythreshold
        isdenses[i, j] = true
        denses[i, j] = Matrix(tile)
        sparses[i, j] = spzeros(T, size(tile)...)
      else
        isdenses[i, j] = false
        denses[i, j] = zeros(T, size(tile)...)
        sparses[i, j] = sparse(tile)
        dropzeros!(sparses[i, j])
      end
    end
    return new{T}(denses, sparses, rowindices, colindices, isempties, isdenses, size(A), Ref(false))
  end
end
function HybridBlockArray(A::AbstractMatrix, ntiles::Integer; sparsitythreshold = 0.2)
  rowindices = collect(chunks(1:size(A, 1); n=ntiles))
  colindices = collect(chunks(1:size(A, 2); n=ntiles))
  return HybridBlockArray(A, rowindices, colindices; sparsitythreshold = sparsitythreshold)
end
Base.size(H::HybridBlockArray) = H.globalsize
Base.size(H::HybridBlockArray, i) = 1 <= i <= 2 ? H.globalsize[i] : 1
Base.eltype(H::HybridBlockArray{T}) where T = T
Base.length(H::HybridBlockArray) = prod(size(H))
function tiles(H, i, j)::Union{Matrix, SparseMatrixCSC}
  return H.isdenses[i, j] ? H.denses[i, j] : H.sparses[i, j]
end
function LinearAlgebra.transpose!(M::Matrix{<:AbstractMatrix})
  for i in eachindex(H.rowindices), j in eachindex(H.colindices)
    if i == j
      M[i, j] .= transpose(M[i, j])
    elseif i > j
      tmp = transpose(M[i, j])
      M[i, j] .= M[j, i]
      M[i, j] .= tmp
    end
  end
end
function LinearAlgebra.transpose!(H::HybridBlockArray{T}) where {T}
  @assert length(H.rowindices) == length(H.colindices)
  H.istransposed[] = !H.istransposed[]
  tmp = H.rowindices
  H.rowindices .= H.colindices
  H.colindices .= tmp
  if H.denses' == H.isdenses'
    transpose!(H.dense)
    transpose!(H.sparse)
  else
    for i in eachindex(H.rowindices), j in eachindex(H.colindices)
      i > j && continue
      # TODO - do this better
      tij = H.isdenses[i, j] ? H.denses[i, j] : H.sparses[i, j]
      tji = H.isdenses[j, i] ? H.denses[j, i] : H.sparses[j, i]
      _tij = deepcopy(tij)
      _tji = deepcopy(tji)
      tij .= transpose(_tji)
      tji .= transpose(_tij)
    end
  end
  H.isempties .= H.isempties'
  return H
end
function Base.axes(H::HybridBlockArray, i)
  i < 0 && throw(BoundsError("axes cannot be called with $i < 1"))
  1 <= i <= 2 && return Base.OneTo(size(H, i))
  return Base.OneTo(1)
end

fastin(i::Integer, r::UnitRange) = r.start <= i <= r.stop
function rowindex(H::HybridBlockArray, i::Int)
  itile = 0
  for (ii, r) in enumerate(H.rowindices)
    fastin(i, r) && (itile = ii; break)
  end
  return itile
end
function colindex(H::HybridBlockArray, j::Int)
  jtile = 0
  for (jj, r) in enumerate(H.colindices)
    fastin(j, r) && (jtile = jj; break)
  end
  return jtile
end
function tileindices(H::HybridBlockArray, i::Int, j::Int)
  return (rowindex(H, i), colindex(H, j))
end
function tileisempty(H::HybridBlockArray, i::Int, j::Int)
  itile, jtile = tileindices(H, i, j)
  return H.isempties[itile, jtile]
end
function tilelocaldindices(H::HybridBlockArray, i::Int, j::Int)
  itile, jtile = tileindices(H, i, j)
  return (i - H.rowindices[itile][1] + 1, j - H.colindices[jtile][1] + 1)
end

function Base.getindex(H::HybridBlockArray{T}, i::Int, j::Int)::T where {T}
  itile, jtile = tileindices(H, i, j)
  if H.isempties[itile, jtile]
    return zero(eltype(H))
  end
  li, lj = tilelocaldindices(H, i, j)
  return tiles(H, itile, jtile)[li, lj]
end
Base.setindex!(H::HybridBlockArray, v, ::Colon, js) = setindex!(H, v, 1:size(H, 1), js)
Base.setindex!(H::HybridBlockArray, v, is, ::Colon) = setindex!(H, v, is, 1:size(H, 2))

function Base.setindex!(H::HybridBlockArray{T}, v::Number, i::Integer, j::Integer) where {T}
#  iszero(v) && return v
  itile, jtile = first.(tileindices(H, i, j))
  li, lj = tilelocaldindices(H, i, j)
  if H.isempties[itile, jtile] && !iszero(v)
    H.isempties[itile, jtile] = false
  end
  if H.isdenses[itile, jtile]
    H.denses[itile, jtile][li, lj] = v
  elseif !iszero(v)
    H.sparses[itile, jtile][li, lj] = v
    H.isempties[itile, jtile] = false
  end
  flushzerotiles!(H)
  return v
end

function Base.setindex!(H::HybridBlockArray{T}, v::AbstractArray{T}, is::AbstractVector{<:Integer}, js::AbstractVector{<:Integer}) where T
  for (ci, i) in enumerate(is), (cj, j) in enumerate(js)
    setindex!(H, v[ci, cj], i, j)
  end
  flushzerotiles!(H)
  return v
end
function flushzerotiles!(H::HybridBlockArray{T}) where T
  for i in eachindex(H.denses, H.sparses)
    d = H.denses[i]
    s = H.sparses[i]
    dropzeros!(s)
    if iszero(d) && iszero(s)
      H.isempties[i] = true
    end
  end
  return H
end

export HybridBlockArray

end # module HybridBlockArrays
