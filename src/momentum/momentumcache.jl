"""
    MomentumCache

Preallocation structure for terms in the momentum equations.
"""
Base.@kwdef struct MomentumCache{T}
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

    u_uz::Vector{T}
    w̄_uz::Vector{T}
    uw̄_uz::Vector{T}

    v_vx::Vector{T}
    ū_vx::Vector{T}
    vū_vx::Vector{T}

    v_vy::Vector{T}
    v̄_vy::Vector{T}
    vv̄_vy::Vector{T}

    v_vz::Vector{T}
    w̄_vz::Vector{T}
    vw̄_vz::Vector{T}

    w_wx::Vector{T}
    ū_wx::Vector{T}
    wū_wx::Vector{T}

    w_wy::Vector{T}
    v̄_wy::Vector{T}
    wv̄_wy::Vector{T}

    w_wz::Vector{T}
    w̄_wz::Vector{T}
    ww̄_wz::Vector{T}

    ∂uū∂x::Vector{T}
    ∂uv̄∂y::Vector{T}
    ∂uw̄∂z::Vector{T}
    ∂vū∂x::Vector{T}
    ∂vv̄∂y::Vector{T}
    ∂vw̄∂z::Vector{T}
    ∂wū∂x::Vector{T}
    ∂wv̄∂y::Vector{T}
    ∂ww̄∂z::Vector{T}

    Conv_ux_11::SparseMatrixCSC{T, Int}
    Conv_uy_11::SparseMatrixCSC{T, Int}
    Conv_uz_11::SparseMatrixCSC{T, Int}
    Conv_uy_12::SparseMatrixCSC{T, Int}
    Conv_uz_13::SparseMatrixCSC{T, Int}

    Conv_vx_21::SparseMatrixCSC{T, Int}
    Conv_vx_22::SparseMatrixCSC{T, Int}
    Conv_vy_22::SparseMatrixCSC{T, Int}
    Conv_vz_22::SparseMatrixCSC{T, Int}
    Conv_vz_23::SparseMatrixCSC{T, Int}

    Conv_wx_31::SparseMatrixCSC{T, Int}
    Conv_wy_32::SparseMatrixCSC{T, Int}
    Conv_wx_33::SparseMatrixCSC{T, Int}
    Conv_wy_33::SparseMatrixCSC{T, Int}
    Conv_wz_33::SparseMatrixCSC{T, Int}
end

function MomentumCache(setup)
    (; Nu, Nv, Nw, NV) = setup.grid
    (; yIu_ux, yIv_uy, yIw_uz, yIu_vx, yIv_vy, yIw_vz, yIu_wx, yIv_wy, yIw_wz) = setup.operators

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

    u_uz = zeros(T, length(yIw_uz))
    w̄_uz = zeros(T, length(yIw_uz))
    uw̄_uz = zeros(T, length(yIw_uz))

    v_vx = zeros(T, length(yIu_vx))
    ū_vx = zeros(T, length(yIu_vx))
    vū_vx = zeros(T, length(yIu_vx))

    v_vy = zeros(T, length(yIv_vy))
    v̄_vy = zeros(T, length(yIv_vy))
    vv̄_vy = zeros(T, length(yIv_vy))

    v_vz = zeros(T, length(yIw_vz))
    w̄_vz = zeros(T, length(yIw_vz))
    vw̄_vz = zeros(T, length(yIw_vz))

    w_wx = zeros(T, length(yIu_wx))
    ū_wx = zeros(T, length(yIu_wx))
    wū_wx = zeros(T, length(yIu_wx))

    w_wy = zeros(T, length(yIv_wy))
    v̄_wy = zeros(T, length(yIv_wy))
    wv̄_wy = zeros(T, length(yIv_wy))

    w_wz = zeros(T, length(yIw_wz))
    w̄_wz = zeros(T, length(yIw_wz))
    ww̄_wz = zeros(T, length(yIw_wz))

    ∂uū∂x = zeros(T, Nu)
    ∂uv̄∂y = zeros(T, Nu)
    ∂uw̄∂z = zeros(T, Nu)
    ∂vū∂x = zeros(T, Nv)
    ∂vv̄∂y = zeros(T, Nv)
    ∂vw̄∂z = zeros(T, Nv)
    ∂wū∂x = zeros(T, Nw)
    ∂wv̄∂y = zeros(T, Nw)
    ∂ww̄∂z = zeros(T, Nw)

    Conv_ux_11  = spzeros(T, Nu, Nu)
    Conv_uy_11 = spzeros(T, Nu, Nu)
    Conv_uz_11 = spzeros(T, Nu, Nu)
    Conv_uy_12 = spzeros(T, Nu, Nv)
    Conv_uz_13 = spzeros(T, Nu, Nw)

    Conv_vx_21 = spzeros(T, Nv, Nu)
    Conv_vx_22 = spzeros(T, Nv, Nv)
    Conv_vy_22 = spzeros(T, Nv, Nv)
    Conv_vz_22 = spzeros(T, Nv, Nv)
    Conv_vz_23 = spzeros(T, Nv, Nw)

    Conv_wx_31 = spzeros(T, Nw, Nu)
    Conv_wy_32 = spzeros(T, Nw, Nv)
    Conv_wx_33 = spzeros(T, Nw, Nw)
    Conv_wy_33 = spzeros(T, Nw, Nw)
    Conv_wz_33 = spzeros(T, Nw, Nw)

    MomentumCache{T}(;
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

        u_uz,
        w̄_uz,
        uw̄_uz,

        v_vx,
        ū_vx,
        vū_vx,

        v_vy,
        v̄_vy,
        vv̄_vy,

        v_vz,
        w̄_vz,
        vw̄_vz,

        w_wx,
        ū_wx,
        wū_wx,

        w_wy,
        v̄_wy,
        wv̄_wy,

        w_wz,
        w̄_wz,
        ww̄_wz,

        ∂uū∂x,
        ∂uv̄∂y,
        ∂uw̄∂z,
        ∂vū∂x,
        ∂vv̄∂y,
        ∂vw̄∂z,
        ∂wū∂x,
        ∂wv̄∂y,
        ∂ww̄∂z,

        Conv_ux_11,
        Conv_uy_11,
        Conv_uz_11,
        Conv_uy_12,
        Conv_uz_13,

        Conv_vx_21,
        Conv_vx_22,
        Conv_vy_22,
        Conv_vz_22,
        Conv_vz_23,

        Conv_wx_31,
        Conv_wy_32,
        Conv_wx_33,
        Conv_wy_33,
        Conv_wz_33,
    )
end

