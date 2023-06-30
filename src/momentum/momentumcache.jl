"""
    MomentumCache(setup)

Preallocation structure for terms in the momentum equations.
"""
MomentumCache(setup) = MomentumCache(setup.grid.dimension, setup)

function MomentumCache(::Dimension{2}, setup)
    (; grid, operators) = setup
    (; Nu, Nv, NV) = grid
    (; Iu_ux, Iv_uy, Iu_vx, Iv_vy) = operators

    T = eltype(Iu_ux)

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

    u_ux = zeros(T, size(Iu_ux, 1))
    ū_ux = zeros(T, size(Iu_ux, 1))
    uū_ux = zeros(T, size(Iu_ux, 1))

    u_uy = zeros(T, size(Iv_uy, 1))
    v̄_uy = zeros(T, size(Iv_uy, 1))
    uv̄_uy = zeros(T, size(Iv_uy, 1))

    v_vx = zeros(T, size(Iu_vx, 1))
    ū_vx = zeros(T, size(Iu_vx, 1))
    vū_vx = zeros(T, size(Iu_vx, 1))

    v_vy = zeros(T, size(Iv_vy, 1))
    v̄_vy = zeros(T, size(Iv_vy, 1))
    vv̄_vy = zeros(T, size(Iv_vy, 1))

    ∂uū∂x = zeros(T, Nu)
    ∂uv̄∂y = zeros(T, Nu)
    ∂vū∂x = zeros(T, Nv)
    ∂vv̄∂y = zeros(T, Nv)

    Conv_ux_11 = spzeros(T, Nu, Nu)
    Conv_uy_11 = spzeros(T, Nu, Nu)
    Conv_uy_12 = spzeros(T, Nu, Nv)

    Conv_vx_21 = spzeros(T, Nv, Nu)
    Conv_vx_22 = spzeros(T, Nv, Nv)
    Conv_vy_22 = spzeros(T, Nv, Nv)

    (;
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
        Conv_vy_22,
    )
end

function MomentumCache(::Dimension{3}, setup)
    (; grid, operators) = setup
    (; Nu, Nv, Nw, NV) = grid
    (; Iu_ux, Iv_uy, Iw_uz, Iu_vx, Iv_vy, Iw_vz, Iu_wx, Iv_wy, Iw_wz) = operators

    T = eltype(Iu_ux)

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

    u_ux = zeros(T, size(Iu_ux, 1))
    ū_ux = zeros(T, size(Iu_ux, 1))
    uū_ux = zeros(T, size(Iu_ux, 1))

    u_uy = zeros(T, size(Iv_uy, 1))
    v̄_uy = zeros(T, size(Iv_uy, 1))
    uv̄_uy = zeros(T, size(Iv_uy, 1))

    u_uz = zeros(T, size(Iw_uz, 1))
    w̄_uz = zeros(T, size(Iw_uz, 1))
    uw̄_uz = zeros(T, size(Iw_uz, 1))

    v_vx = zeros(T, size(Iu_vx, 1))
    ū_vx = zeros(T, size(Iu_vx, 1))
    vū_vx = zeros(T, size(Iu_vx, 1))

    v_vy = zeros(T, size(Iv_vy, 1))
    v̄_vy = zeros(T, size(Iv_vy, 1))
    vv̄_vy = zeros(T, size(Iv_vy, 1))

    v_vz = zeros(T, size(Iw_vz, 1))
    w̄_vz = zeros(T, size(Iw_vz, 1))
    vw̄_vz = zeros(T, size(Iw_vz, 1))

    w_wx = zeros(T, size(Iu_wx, 1))
    ū_wx = zeros(T, size(Iu_wx, 1))
    wū_wx = zeros(T, size(Iu_wx, 1))

    w_wy = zeros(T, size(Iv_wy, 1))
    v̄_wy = zeros(T, size(Iv_wy, 1))
    wv̄_wy = zeros(T, size(Iv_wy, 1))

    w_wz = zeros(T, size(Iw_wz, 1))
    w̄_wz = zeros(T, size(Iw_wz, 1))
    ww̄_wz = zeros(T, size(Iw_wz, 1))

    ∂uū∂x = zeros(T, Nu)
    ∂uv̄∂y = zeros(T, Nu)
    ∂uw̄∂z = zeros(T, Nu)
    ∂vū∂x = zeros(T, Nv)
    ∂vv̄∂y = zeros(T, Nv)
    ∂vw̄∂z = zeros(T, Nv)
    ∂wū∂x = zeros(T, Nw)
    ∂wv̄∂y = zeros(T, Nw)
    ∂ww̄∂z = zeros(T, Nw)

    Conv_ux_11 = spzeros(T, Nu, Nu)
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

    (;
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
