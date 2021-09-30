struct MomentumCache{T}
    Gp::Vector{T}
    c::Vector{T}
    c2::Vector{T}
    c3::Vector{T}
    d::Vector{T}
    b::Vector{T}
    ∇c::SparseMatrixCSC{T,Int}
    ∇c2::SparseMatrixCSC{T,Int}
    ∇c3::SparseMatrixCSC{T,Int}
    ∇d::SparseMatrixCSC{T,Int}
    ∇b::SparseMatrixCSC{T,Int}
end
function MomentumCache(setup::Setup{T}) where {T}
    @unpack NV = setup.grid
    Gp = zeros(T, NV)
    c = zeros(T, NV)
    c2 = zeros(T, NV)
    c3 = zeros(T, NV)
    d = zeros(T, NV)
    b = zeros(T, NV)
    ∇c = spzeros(T, NV, NV)
    ∇c2 = spzeros(T, NV, NV)
    ∇c3 = spzeros(T, NV, NV)
    ∇d = spzeros(T, NV, NV)
    ∇b = spzeros(T, NV, NV)
    MomentumCache{T}(Gp, c, c2, c3, d, b, ∇c, ∇c2, ∇c3, ∇d, ∇b)
end
