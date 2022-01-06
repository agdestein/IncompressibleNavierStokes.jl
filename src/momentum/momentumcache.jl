"""
    MomentumCache

Preallocation structure for terms in the momentum equations.
"""
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

    u_ux::Vector{T}
    ū_ux::Vector{T}
    uū_ux::Vector{T}

    u_uy::Vector{T}
    v̄_uy::Vector{T}
    uv̄_uy::Vector{T}

    v_vx::Vector{T}
    ū_vx::Vector{T}
    vū_vx::Vector{T}

    v_vy::Vector{T}
    v̄_vy::Vector{T}
    vv̄_vy::Vector{T}

    ∂uū∂x::Vector{T}
    ∂uv̄∂y::Vector{T}
    ∂vū∂x::Vector{T}
    ∂vv̄∂y::Vector{T}

    Conv_ux_11::SparseMatrixCSC{T, Int}
    Conv_uy_11::SparseMatrixCSC{T, Int}
    Conv_uy_12::SparseMatrixCSC{T, Int}
    Conv_vx_21::SparseMatrixCSC{T, Int}
    Conv_vx_22::SparseMatrixCSC{T, Int}
    Conv_vy_22::SparseMatrixCSC{T, Int}
end
function MomentumCache(setup)
    (; Nu, Nv, NV) = setup.grid
    (; yIu_ux, yIv_uy, yIu_vx, yIv_vy) = setup.operators

    T = eltype(yIu_ux)

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

    u_ux = zeros(T, length(yIu_ux))
    ū_ux = zeros(T, length(yIu_ux))
    uū_ux = zeros(T, length(yIu_ux))

    u_uy = zeros(T, length(yIv_uy))
    v̄_uy = zeros(T, length(yIv_uy))
    uv̄_uy = zeros(T, length(yIv_uy))

    v_vx = zeros(T, length(yIu_vx))
    ū_vx = zeros(T, length(yIu_vx))
    vū_vx = zeros(T, length(yIu_vx))

    v_vy = zeros(T, length(yIv_vy))
    v̄_vy = zeros(T, length(yIv_vy))
    vv̄_vy = zeros(T, length(yIv_vy))

    ∂uū∂x = zeros(T, Nu)
    ∂uv̄∂y = zeros(T, Nu)
    ∂vū∂x = zeros(T, Nv)
    ∂vv̄∂y = zeros(T, Nv)

    Conv_ux_11  = spzeros(T, Nu, Nu)

    Conv_uy_11 = spzeros(T, Nu, Nv)
    Conv_uy_12 = spzeros(T, Nu, Nv)

    Conv_vx_21 = spzeros(T, Nv, Nu)
    Conv_vx_22 = spzeros(T, Nv, Nu)

    Conv_vy_22 = spzeros(T, Nv, Nv)

    MomentumCache{T}(
        Gp,
        c,
        c2,
        c3,
        d,
        b,
        ∇c,
        ∇c2,
        ∇c3,
        ∇d,
        ∇b,
        u_ux,
        ū_ux,
        uū_ux,
        u_uy,
        v̄_uy,
        uv̄_uy,
        v_vx,
        ū_vx,
        vū_vx,
        v_vy,
        v̄_vy,
        vv̄_vy,
        ∂uū∂x,
        ∂uv̄∂y,
        ∂vū∂x,
        ∂vv̄∂y,
        Conv_ux_11,
        Conv_uy_11,
        Conv_uy_12,
        Conv_vx_21,
        Conv_vx_22,
        Conv_vy_22
    )
end
