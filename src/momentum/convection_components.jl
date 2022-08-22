"""
    convection_components(
        V, ϕ, setup;
        getJacobian = false,
        newton_factor = false,
        order4 = false,
    )

Compute convection components.

Non-mutating/allocating/out-of-place version.

See also [`convection_components!`](@ref).
"""
function convection_components end

# 2D version
function convection_components(
    V,
    ϕ,
    setup::Setup{T,2};
    getJacobian = false,
    newton_factor = false,
    order4 = false,
) where {T}
    (; grid, operators) = setup

    if order4
        Cux = operators.Cux3
        Cuy = operators.Cuy3
        Cvx = operators.Cvx3
        Cvy = operators.Cvy3

        Au_ux = operators.Au_ux3
        Au_uy = operators.Au_uy3
        Av_vx = operators.Av_vx3
        Av_vy = operators.Av_vy3

        yAu_ux = operators.yAu_ux3
        yAu_uy = operators.yAu_uy3
        yAv_vx = operators.yAv_vx3
        yAv_vy = operators.yAv_vy3

        Iu_ux = operators.Iu_ux3
        Iv_uy = operators.Iv_uy3
        Iu_vx = operators.Iu_vx3
        Iv_vy = operators.Iv_vy3

        yIu_ux = operators.yIu_ux3
        yIv_uy = operators.yIv_uy3
        yIu_vx = operators.yIu_vx3
        yIv_vy = operators.yIv_vy3
    else
        (; Cux, Cuy, Cvx, Cvy) = operators
        (; Au_ux, Au_uy, Av_vx, Av_vy) = operators
        (; yAu_ux, yAu_uy, yAv_vx, yAv_vy) = operators
        (; Iu_ux, Iv_uy, Iu_vx, Iv_vy) = operators
        (; yIu_ux, yIv_uy, yIu_vx, yIv_vy) = operators
    end

    (; indu, indv) = grid

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # Convection components
    u_ux = Au_ux * uₕ + yAu_ux                # u at ux
    ū_ux = Iu_ux * ϕu + yIu_ux                # ū at ux
    ∂uū∂x = Cux * (u_ux .* ū_ux)

    u_uy = Au_uy * uₕ + yAu_uy                # u at uy
    v̄_uy = Iv_uy * ϕv + yIv_uy                # ū at uy
    ∂uv̄∂y = Cuy * (u_uy .* v̄_uy)

    v_vx = Av_vx * vₕ + yAv_vx                # v at vx
    ū_vx = Iu_vx * ϕu + yIu_vx                # ū at vx
    ∂vū∂x = Cvx * (v_vx .* ū_vx)

    v_vy = Av_vy * vₕ + yAv_vy                # v at vy
    v̄_vy = Iv_vy * ϕv + yIv_vy                # ū at vy
    ∂vv̄∂y = Cvy * (v_vy .* v̄_vy)

    cu = ∂uū∂x + ∂uv̄∂y
    cv = ∂vū∂x + ∂vv̄∂y

    c = [cu; cv]

    if getJacobian
        ## Convective terms, u-component
        C1 = Cux * Diagonal(ū_ux)
        C2 = Cux * Diagonal(u_ux) * newton_factor
        Conv_ux_11 = C1 * Au_ux .+ C2 * Iu_ux

        C1 = Cuy * Diagonal(v̄_uy)
        C2 = Cuy * Diagonal(u_uy) * newton_factor
        Conv_uy_11 = C1 * Au_uy
        Conv_uy_12 = C2 * Iv_uy

        ## Convective terms, v-component
        C1 = Cvx * Diagonal(ū_vx)
        C2 = Cvx * Diagonal(v_vx) * newton_factor
        Conv_vx_21 = C2 * Iu_vx
        Conv_vx_22 = C1 * Av_vx

        C1 = Cvy * Diagonal(v̄_vy)
        C2 = Cvy * Diagonal(v_vy) * newton_factor
        Conv_vy_22 = C1 * Av_vy .+ C2 * Iv_vy

        ∇c = [
            (Conv_ux_11 + Conv_uy_11) Conv_uy_12
            Conv_vx_21 (Conv_vx_22 + Conv_vy_22)
        ]
    end

    c, ∇c
end

# 3D version
function convection_components(
        V,
        ϕ,
        setup::Setup{T,3};
        getJacobian = false,
        newton_factor = false,
        order4 = false,
) where {T}
    order4 && error("order4 not implemented for 3D")

    (; grid, operators) = setup
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = operators
    (; Au_ux, Au_uy, Au_uz) = operators
    (; Av_vx, Av_vy, Av_vz) = operators
    (; Aw_wx, Aw_wy, Aw_wz) = operators
    (; yAu_ux, yAu_uy, yAu_uz) = operators
    (; yAv_vx, yAv_vy, yAv_vz) = operators
    (; yAw_wx, yAw_wy, yAw_wz) = operators
    (; Iu_ux, Iv_uy, Iw_uz) = operators
    (; Iu_vx, Iv_vy, Iw_vz) = operators
    (; Iu_wx, Iv_wy, Iw_wz) = operators
    (; yIu_ux, yIv_uy, yIw_uz) = operators
    (; yIu_vx, yIv_vy, yIw_vz) = operators
    (; yIu_wx, yIv_wy, yIw_wz) = operators
    (; indu, indv, indw) = grid

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    u_ux = Au_ux * uₕ + yAu_ux                # u at ux
    ū_ux = Iu_ux * ϕu + yIu_ux                # ū at ux
    ∂uū∂x = Cux * (u_ux .* ū_ux)

    u_uy = Au_uy * uₕ + yAu_uy                # u at uy
    v̄_uy = Iv_uy * ϕv + yIv_uy                # v̄ at uy
    ∂uv̄∂y = Cuy * (u_uy .* v̄_uy)

    u_uz = Au_uz * uₕ + yAu_uz                # u at uz
    w̄_uz = Iw_uz * ϕw + yIw_uz                # ū at uz
    ∂uw̄∂z = Cuz * (u_uz .* w̄_uz)

    v_vx = Av_vx * vₕ + yAv_vx                # v at vx
    ū_vx = Iu_vx * ϕu + yIu_vx                # ū at vx
    ∂vū∂x = Cvx * (v_vx .* ū_vx)

    v_vy = Av_vy * vₕ + yAv_vy                # v at vy
    v̄_vy = Iv_vy * ϕv + yIv_vy                # v̄ at vy
    ∂vv̄∂y = Cvy * (v_vy .* v̄_vy)

    v_vz = Av_vz * vₕ + yAv_vz                # v at vz
    w̄_vz = Iw_vz * ϕw + yIw_vz                # w̄ at vz
    ∂vw̄∂z = Cvz * (v_vz .* w̄_vz)

    w_wx = Aw_wx * wₕ + yAw_wx                # w at wx
    ū_wx = Iu_wx * ϕu + yIu_wx                # ū at wx
    ∂wū∂x = Cwx * (w_wx .* ū_wx)

    w_wy = Aw_wy * wₕ + yAw_wy                # w at wy
    v̄_wy = Iv_wy * ϕv + yIv_wy                # v̄ at wy
    ∂wv̄∂y = Cwy * (w_wy .* v̄_wy)

    w_wz = Aw_wz * wₕ + yAw_wz                # w at wz
    w̄_wz = Iw_wz * ϕw + yIw_wz                # w̄ at wz
    ∂ww̄∂z = Cwz * (w_wz .* w̄_wz)

    cu = @. ∂uū∂x + ∂uv̄∂y + ∂uw̄∂z
    cv = @. ∂vū∂x + ∂vv̄∂y + ∂vw̄∂z
    cw = @. ∂wū∂x + ∂wv̄∂y + ∂ww̄∂z

    c = [cu; cv; cw]

    if getJacobian
        ## Convective terms, u-component
        C1 = Cux * Diagonal(ū_ux)
        C2 = Cux * Diagonal(u_ux) * newton_factor
        Conv_ux_11 = C1 * Au_ux .+ C2 * Iu_ux

        C1 = Cuy * Diagonal(v̄_uy)
        C2 = Cuy * Diagonal(u_uy) * newton_factor
        Conv_uy_11 = C1 * Au_uy
        Conv_uy_12 = C2 * Iv_uy

        C1 = Cuz * Diagonal(w̄_uz)
        C2 = Cuz * Diagonal(u_uz) * newton_factor
        Conv_uz_11 = C1 * Au_uz
        Conv_uz_13 = C2 * Iw_uz

        ## Convective terms, v-component
        C1 = Cvx * Diagonal(ū_vx)
        C2 = Cvx * Diagonal(v_vx) * newton_factor
        Conv_vx_21 = C2 * Iu_vx
        Conv_vx_22 = C1 * Av_vx

        C1 = Cvy * Diagonal(v̄_vy)
        C2 = Cvy * Diagonal(v_vy) * newton_factor
        Conv_vy_22 = C1 * Av_vy .+ C2 * Iv_vy

        C1 = Cvz * Diagonal(w̄_vz)
        C2 = Cvz * Diagonal(v_vz) * newton_factor
        Conv_vz_23 = C2 * Iw_vz
        Conv_vz_22 = C1 * Av_vz

        ## Convective terms, w-component
        C1 = Cwx * Diagonal(ū_wx)
        C2 = Cwx * Diagonal(w_wx) * newton_factor
        Conv_wx_31 = C2 * Iu_wx
        Conv_wx_33 = C1 * Aw_wx

        C1 = Cwy * Diagonal(v̄_wy)
        C2 = Cwy * Diagonal(w_wy) * newton_factor
        Conv_wy_32 = C2 * Iv_wy
        Conv_wy_33 = C1 * Aw_wy

        C1 = Cwz * Diagonal(w̄_wz)
        C2 = Cwz * Diagonal(w_wz) * newton_factor
        Conv_wz_33 = C1 * Aw_wz .+ C2 * Iw_wz

        ## Jacobian
        ∇c = [
            (Conv_ux_11 + Conv_uy_11 + Conv_uz_11) Conv_uy_12 Conv_uz_13
            Conv_vx_21 (Conv_vx_22 + Conv_vy_22 + Conv_vz_22) Conv_vz_23
            Conv_wx_31 Conv_wy_32 (Conv_wx_33 + Conv_wy_33 + Conv_wz_33)
        ]
    end

    c, ∇c
end

"""
    convection_components!(
        c, ∇c, V, ϕ, setup, cache;
        getJacobian = false,
        newton_factor = false,
        order4 = false,
    )

Compute convection components.

Mutating/non-allocating/in-place version.

See also [`convection_components`](@ref).
"""
function convection_components! end

# 2D version
function convection_components!(
        c,
        ∇c,
        V,
        ϕ,
        setup::Setup{T,2},
        cache;
        getJacobian = false,
        newton_factor = false,
        order4 = false,
) where {T}
    (; grid, operators) = setup

    if order4
        Cux = operators.Cux3
        Cuy = operators.Cuy3
        Cvx = operators.Cvx3
        Cvy = operators.Cvy3

        Au_ux = operators.Au_ux3
        Au_uy = operators.Au_uy3
        Av_vx = operators.Av_vx3
        Av_vy = operators.Av_vy3

        yAu_ux = operators.yAu_ux3
        yAu_uy = operators.yAu_uy3
        yAv_vx = operators.yAv_vx3
        yAv_vy = operators.yAv_vy3

        Iu_ux = operators.Iu_ux3
        Iv_uy = operators.Iv_uy3
        Iu_vx = operators.Iu_vx3
        Iv_vy = operators.Iv_vy3

        yIu_ux = operators.yIu_ux3
        yIv_uy = operators.yIv_uy3
        yIu_vx = operators.yIu_vx3
        yIv_vy = operators.yIv_vy3
    else
        (; Cux, Cuy, Cvx, Cvy) = operators
        (; Au_ux, Au_uy, Av_vx, Av_vy) = operators
        (; yAu_ux, yAu_uy, yAv_vx, yAv_vy) = operators
        (; Iu_ux, Iv_uy, Iu_vx, Iv_vy) = operators
        (; yIu_ux, yIv_uy, yIu_vx, yIv_vy) = operators
    end

    (; indu, indv) = grid

    (; u_ux, ū_ux, uū_ux, u_uy, v̄_uy, uv̄_uy) = cache
    (; v_vx, ū_vx, vū_vx, v_vy, v̄_vy, vv̄_vy) = cache

    (; ∂uū∂x, ∂uv̄∂y, ∂vū∂x, ∂vv̄∂y) = cache
    (; Conv_ux_11, Conv_uy_11, Conv_uy_12, Conv_vx_21, Conv_vx_22, Conv_vy_22) = cache

    cu = @view c[indu]
    cv = @view c[indv]

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # Convection components
    if order4
        # TODO: preallocated arrays for order4
        u_ux = Au_ux * uₕ + yAu_ux                # U at ux
        ū_ux = Iu_ux * ϕu + yIu_ux                # Ū at ux
        # ∂uū∂x = Cux * (u_ux .* ū_ux)
        uū_ux = @. u_ux = u_ux * ū_ux

        u_uy = Au_uy * uₕ + yAu_uy                # U at uy
        v̄_uy = Iv_uy * ϕv + yIv_uy                # Ū at uy
        # ∂uv̄∂y = Cuy * (u_uy .* v̄_uy)
        uv̄_uy = @. u_uy = u_uy * v̄_uy

        v_vx = Av_vx * vₕ + yAv_vx                # V at vx
        ū_vx = Iu_vx * ϕu + yIu_vx                # Ū at vx
        ∂vū∂x = Cvx * (v_vx .* ū_vx)
        vū_vx = @. v_vx = v_vx * ū_vx

        v_vy = Av_vy * vₕ + yAv_vy                # V at vy
        v̄_vy = Iv_vy * ϕv + yIv_vy                # Ū at vy
        # ∂vv̄∂y = Cvy * (v_vy .* v̄_vy)
        vv̄_vy = @. v_vy = v_vy * v̄_vy
    else
        # Fill preallocated arrays
        mul!(u_ux, Au_ux, uₕ)
        mul!(ū_ux, Iu_ux, ϕu)
        mul!(u_uy, Au_uy, uₕ)
        mul!(v̄_uy, Iv_uy, ϕv)
        mul!(v_vx, Av_vx, vₕ)
        mul!(ū_vx, Iu_vx, ϕu)
        mul!(v_vy, Av_vy, vₕ)
        mul!(v̄_vy, Iv_vy, ϕv)

        u_ux .+= yAu_ux
        ū_ux .+= yIu_ux
        @. uū_ux = u_ux * ū_ux

        u_uy .+= yAu_uy
        v̄_uy .+= yIv_uy
        @. uv̄_uy = u_uy * v̄_uy

        v_vx .+= yAv_vx
        ū_vx .+= yIu_vx
        @. vū_vx = v_vx * ū_vx

        v_vy .+= yAv_vy
        v̄_vy .+= yIv_vy
        @. vv̄_vy = v_vy * v̄_vy
    end

    mul!(∂uū∂x, Cux, uū_ux)
    mul!(∂uv̄∂y, Cuy, uv̄_uy)
    mul!(∂vū∂x, Cvx, vū_vx)
    mul!(∂vv̄∂y, Cvy, vv̄_vy)

    @. cu = ∂uū∂x + ∂uv̄∂y
    @. cv = ∂vū∂x + ∂vv̄∂y

    if getJacobian
        ## Convective terms, u-component
        C1 = Cux * Diagonal(ū_ux)
        C2 = Cux * Diagonal(u_ux) * newton_factor
        Conv_ux_11 .= C1 * Au_ux .+ C2 * Iu_ux

        C1 = Cuy * Diagonal(v̄_uy)
        C2 = Cuy * Diagonal(u_uy) * newton_factor
        Conv_uy_11 .= C1 * Au_uy
        Conv_uy_12 .= C2 * Iv_uy

        ## Convective terms, v-component
        C1 = Cvx * Diagonal(ū_vx)
        C2 = Cvx * Diagonal(v_vx) * newton_factor
        Conv_vx_21 .= C2 * Iu_vx
        Conv_vx_22 .= C1 * Av_vx

        C1 = Cvy * Diagonal(v̄_vy)
        C2 = Cvy * Diagonal(v_vy) * newton_factor
        Conv_vy_22 .= C1 * Av_vy .+ C2 * Iv_vy

        ∇c[indu, indu] = Conv_ux_11 + Conv_uy_11
        ∇c[indu, indv] = Conv_uy_12
        ∇c[indv, indu] = Conv_vx_21
        ∇c[indv, indv] = Conv_vx_22 + Conv_vy_22
    end

    c, ∇c
end

# 3D version
function convection_components!(
        c,
        ∇c,
        V,
        ϕ,
        setup::Setup{T,3},
        cache;
        getJacobian = false,
        newton_factor = false,
        order4 = false,
) where {T}
    order4 && error("order4 not implemented for 3D")

    (; grid, operators) = setup
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = operators
    (; Au_ux, Au_uy, Au_uz) = operators
    (; Av_vx, Av_vy, Av_vz) = operators
    (; Aw_wx, Aw_wy, Aw_wz) = operators
    (; yAu_ux, yAu_uy, yAu_uz) = operators
    (; yAv_vx, yAv_vy, yAv_vz) = operators
    (; yAw_wx, yAw_wy, yAw_wz) = operators
    (; Iu_ux, Iv_uy, Iw_uz) = operators
    (; Iu_vx, Iv_vy, Iw_vz) = operators
    (; Iu_wx, Iv_wy, Iw_wz) = operators
    (; yIu_ux, yIv_uy, yIw_uz) = operators
    (; yIu_vx, yIv_vy, yIw_vz) = operators
    (; yIu_wx, yIv_wy, yIw_wz) = operators
    (; indu, indv, indw) = grid
    (; u_ux, ū_ux, uū_ux, u_uy, v̄_uy, uv̄_uy, u_uz, w̄_uz, uw̄_uz) = cache
    (; v_vx, ū_vx, vū_vx, v_vy, v̄_vy, vv̄_vy, v_vz, w̄_vz, vw̄_vz) = cache
    (; w_wx, ū_wx, wū_wx, w_wy, v̄_wy, wv̄_wy, w_wz, w̄_wz, ww̄_wz) = cache
    (; ∂uū∂x, ∂uv̄∂y, ∂uw̄∂z, ∂vū∂x, ∂vv̄∂y, ∂vw̄∂z, ∂wū∂x, ∂wv̄∂y, ∂ww̄∂z) = cache
    (; Conv_ux_11, Conv_uy_11, Conv_uz_11, Conv_uy_12, Conv_uz_13) = cache
    (; Conv_vx_21, Conv_vx_22, Conv_vy_22, Conv_vz_22, Conv_vz_23) = cache
    (; Conv_wx_31, Conv_wy_32, Conv_wx_33, Conv_wy_33, Conv_wz_33) = cache

    cu = @view c[indu]
    cv = @view c[indv]
    cw = @view c[indw]

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    # Convection components
    mul!(u_ux, Au_ux, uₕ)
    mul!(ū_ux, Iu_ux, ϕu)
    mul!(u_uy, Au_uy, uₕ)
    mul!(v̄_uy, Iv_uy, ϕv)
    mul!(u_uz, Au_uz, uₕ)
    mul!(w̄_uz, Iw_uz, ϕw)

    mul!(v_vx, Av_vx, vₕ)
    mul!(ū_vx, Iu_vx, ϕu)
    mul!(v_vy, Av_vy, vₕ)
    mul!(v̄_vy, Iv_vy, ϕv)
    mul!(v_vz, Av_vz, vₕ)
    mul!(w̄_vz, Iw_vz, ϕw)

    mul!(w_wx, Aw_wx, wₕ)
    mul!(ū_wx, Iu_wx, ϕu)
    mul!(w_wy, Aw_wy, wₕ)
    mul!(v̄_wy, Iv_wy, ϕv)
    mul!(w_wz, Aw_wz, wₕ)
    mul!(w̄_wz, Iw_wz, ϕw)

    u_ux .+= yAu_ux
    ū_ux .+= yIu_ux
    @. uū_ux = u_ux * ū_ux

    u_uy .+= yAu_uy
    v̄_uy .+= yIv_uy
    @. uv̄_uy = u_uy * v̄_uy

    u_uz .+= yAu_uz
    w̄_uz .+= yIw_uz
    @. uw̄_uz = u_uz * w̄_uz

    v_vx .+= yAv_vx
    ū_vx .+= yIu_vx
    @. vū_vx = v_vx * ū_vx

    v_vy .+= yAv_vy
    v̄_vy .+= yIv_vy
    @. vv̄_vy = v_vy * v̄_vy

    v_vz .+= yAv_vz
    w̄_vz .+= yIw_vz
    @. vw̄_vz = v_vz * w̄_vz

    w_wx .+= yAw_wx
    ū_wx .+= yIu_wx
    @. wū_wx = w_wx * ū_wx

    w_wy .+= yAw_wy
    v̄_wy .+= yIv_wy
    @. wv̄_wy = w_wy * v̄_wy

    w_wz .+= yAw_wz
    w̄_wz .+= yIw_wz
    @. ww̄_wz = w_wz * w̄_wz

    mul!(∂uū∂x, Cux, uū_ux)
    mul!(∂uv̄∂y, Cuy, uv̄_uy)
    mul!(∂uw̄∂z, Cuz, uw̄_uz)

    mul!(∂vū∂x, Cvx, vū_vx)
    mul!(∂vv̄∂y, Cvy, vv̄_vy)
    mul!(∂vw̄∂z, Cvz, vw̄_vz)

    mul!(∂wū∂x, Cwx, wū_wx)
    mul!(∂wv̄∂y, Cwy, wv̄_wy)
    mul!(∂ww̄∂z, Cwz, ww̄_wz)

    @. cu = ∂uū∂x + ∂uv̄∂y + ∂uw̄∂z
    @. cv = ∂vū∂x + ∂vv̄∂y + ∂vw̄∂z
    @. cw = ∂wū∂x + ∂wv̄∂y + ∂ww̄∂z

    if getJacobian
        ## Convective terms, u-component
        C1 = Cux * Diagonal(ū_ux)
        C2 = Cux * Diagonal(u_ux) * newton_factor
        Conv_ux_11 .= C1 * Au_ux .+ C2 * Iu_ux

        C1 = Cuy * Diagonal(v̄_uy)
        C2 = Cuy * Diagonal(u_uy) * newton_factor
        Conv_uy_11 .= C1 * Au_uy
        Conv_uy_12 .= C2 * Iv_uy

        C1 = Cuz * Diagonal(w̄_uz)
        C2 = Cuz * Diagonal(u_uz) * newton_factor
        Conv_uz_11 .= C1 * Au_uz
        Conv_uz_13 .= C2 * Iw_uz

        ## Convective terms, v-component
        C1 = Cvx * Diagonal(ū_vx)
        C2 = Cvx * Diagonal(v_vx) * newton_factor
        Conv_vx_21 .= C2 * Iu_vx
        Conv_vx_22 .= C1 * Av_vx

        C1 = Cvy * Diagonal(v̄_vy)
        C2 = Cvy * Diagonal(v_vy) * newton_factor
        Conv_vy_22 .= C1 * Av_vy .+ C2 * Iv_vy

        C1 = Cvz * Diagonal(w̄_vz)
        C2 = Cvz * Diagonal(v_vz) * newton_factor
        Conv_vz_23 .= C2 * Iw_vz
        Conv_vz_22 .= C1 * Av_vz

        ## Convective terms, w-component
        C1 = Cwx * Diagonal(ū_wx)
        C2 = Cwx * Diagonal(w_wx) * newton_factor
        Conv_wx_31 .= C2 * Iu_wx
        Conv_wx_33 .= C1 * Aw_wx

        C1 = Cwy * Diagonal(v̄_wy)
        C2 = Cwy * Diagonal(w_wy) * newton_factor
        Conv_wy_32 .= C2 * Iv_wy
        Conv_wy_33 .= C1 * Aw_wy

        C1 = Cwz * Diagonal(w̄_wz)
        C2 = Cwz * Diagonal(w_wz) * newton_factor
        Conv_wz_33 .= C1 * Aw_wz .+ C2 * Iw_wz

        ## Jacobian
        ∇c[indu, indu] = Conv_ux_11 + Conv_uy_11 + Conv_uz_11
        ∇c[indu, indv] = Conv_uy_12
        ∇c[indu, indw] = Conv_uz_13
        ∇c[indv, indu] = Conv_vx_21
        ∇c[indv, indv] = Conv_vx_22 + Conv_vy_22 + Conv_vz_22
        ∇c[indv, indw] = Conv_vz_23
        ∇c[indw, indu] = Conv_wx_31
        ∇c[indw, indv] = Conv_wy_32
        ∇c[indw, indw] = Conv_wx_33 + Conv_wy_33 + Conv_wz_33
    end

    c, ∇c
end
