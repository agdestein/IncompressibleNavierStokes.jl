"""
    convection(V, setup)

Compute convective forces (out-of-place version).
"""
function convection end

# 2D version
function convection(V, setup::Setup{T,2}) where {T}
    (; grid, operators) = setup
    (; Cux, Cuy, Cvx, Cvy) = operators
    (; Au_ux, Au_uy, Av_vx, Av_vy) = operators
    (; Iu_ux, Iv_uy, Iu_vx, Iv_vy) = operators
    (; indu, indv) = grid

    u = @view V[indu]
    v = @view V[indv]

    ∂uu∂x = Cux * ((Au_ux * u) .* (Iu_ux * u))
    ∂uv∂y = Cuy * ((Au_uy * u) .* (Iv_uy * v))
    ∂vu∂x = Cvx * ((Av_vx * v) .* (Iu_vx * u))
    ∂vv∂y = Cvy * ((Av_vy * v) .* (Iv_vy * v))

    cu = ∂uu∂x + ∂uv∂y
    cv = ∂vu∂x + ∂vv∂y

    [cu; cv]
end

# 3D version
function convection(V, setup::Setup{T,3}) where {T}
    (; grid, operators) = setup
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = operators
    (; Au_ux, Au_uy, Au_uz) = operators
    (; Av_vx, Av_vy, Av_vz) = operators
    (; Aw_wx, Aw_wy, Aw_wz) = operators
    (; Iu_ux, Iv_uy, Iw_uz) = operators
    (; Iu_vx, Iv_vy, Iw_vz) = operators
    (; Iu_wx, Iv_wy, Iw_wz) = operators
    (; indu, indv, indw) = grid

    u = @view V[indu]
    v = @view V[indv]
    w = @view V[indw]

    ∂uu∂x = Cux * ((Au_ux * u) .* (Iu_ux * u))
    ∂uv∂y = Cuy * ((Au_uy * u) .* (Iv_uy * v))
    ∂uw∂z = Cuz * ((Au_uz * u) .* (Iw_uz * w))

    ∂vu∂x = Cvx * ((Av_vx * v) .* (Iu_vx * u))
    ∂vv∂y = Cvy * ((Av_vy * v) .* (Iv_vy * v))
    ∂vw∂z = Cvz * ((Av_vz * v) .* (Iw_vz * w))

    ∂wu∂x = Cwx * ((Aw_wx * w) .* (Iu_wx * u))
    ∂wv∂y = Cwy * ((Aw_wy * w) .* (Iv_wy * v))
    ∂ww∂z = Cwz * ((Aw_wz * w) .* (Iw_wz * w))

    cu = ∂uu∂x + ∂uv∂y + ∂uw∂z
    cv = ∂vu∂x + ∂vv∂y + ∂vw∂z
    cw = ∂wu∂x + ∂wv∂y + ∂ww∂z

    [cu; cv; cw]
end


"""
    convection!(c, V, setup, cache)

Compute convective forces (in-place mutating version).
"""
function convection! end

# 2D version
function convection!(c, V, setup::Setup{T,2}, cache) where {T}
    (; grid, operators) = setup

    (; Cux, Cuy, Cvx, Cvy) = operators
    (; Au_ux, Au_uy, Av_vx, Av_vy) = operators
    (; Iu_ux, Iv_uy, Iu_vx, Iv_vy) = operators

    (; indu, indv) = grid

        (; u_ux, ū_ux, uū_ux, u_uy, v̄_uy, uv̄_uy) = cache
        (; v_vx, ū_vx, vū_vx, v_vy, v̄_vy, vv̄_vy) = cache

        (; ∂uū∂x, ∂uv̄∂y, ∂vū∂x, ∂vv̄∂y) = cache

    cu = @view c[indu]
    cv = @view c[indv]

    u = @view V[indu]
    v = @view V[indv]

    # Fill preallocated arrays
    mul!(u_ux, Au_ux, u)
     mul!(ū_ux, Iu_ux, u)
    mul!(u_uy, Au_uy, u)
     mul!(v̄_uy, Iv_uy, v)
    mul!(v_vx, Av_vx, v)
     mul!(ū_vx, Iu_vx, u)
    mul!(v_vy, Av_vy, v)
     mul!(v̄_vy, Iv_vy, v)

      @. uū_ux = u_ux * ū_ux
      @. uv̄_uy = u_uy * v̄_uy
      @. vū_vx = v_vx * ū_vx
      @. vv̄_vy = v_vy * v̄_vy

      mul!(∂uū∂x, Cux, uū_ux)
      mul!(∂uv̄∂y, Cuy, uv̄_uy)
      mul!(∂vū∂x, Cvx, vū_vx)
      mul!(∂vv̄∂y, Cvy, vv̄_vy)

      @. cu = ∂uū∂x + ∂uv̄∂y
      @. cv = ∂vū∂x + ∂vv̄∂y

    c
end

# 3D version
function convection!(c, V, setup::Setup{T,3}, cache) where {T}
    (; grid, operators) = setup
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = operators
    (; Au_ux, Au_uy, Au_uz) = operators
    (; Av_vx, Av_vy, Av_vz) = operators
    (; Aw_wx, Aw_wy, Aw_wz) = operators
    (; Iu_ux, Iv_uy, Iw_uz) = operators
    (; Iu_vx, Iv_vy, Iw_vz) = operators
    (; Iu_wx, Iv_wy, Iw_wz) = operators
    (; indu, indv, indw) = grid
          (; u_ux, ū_ux, uū_ux, u_uy, v̄_uy, uv̄_uy, u_uz, w̄_uz, uw̄_uz) = cache
          (; v_vx, ū_vx, vū_vx, v_vy, v̄_vy, vv̄_vy, v_vz, w̄_vz, vw̄_vz) = cache
          (; w_wx, ū_wx, wū_wx, w_wy, v̄_wy, wv̄_wy, w_wz, w̄_wz, ww̄_wz) = cache
             (; ∂uū∂x, ∂uv̄∂y, ∂uw̄∂z, ∂vū∂x, ∂vv̄∂y, ∂vw̄∂z, ∂wū∂x, ∂wv̄∂y, ∂ww̄∂z) = cache

    cu = @view c[indu]
    cv = @view c[indv]
    cw = @view c[indw]

    u = @view V[indu]
    v = @view V[indv]
    w = @view V[indw]

    # Convection components
    mul!(u_ux, Au_ux, u)
     mul!(ū_ux, Iu_ux, u)
    mul!(u_uy, Au_uy, u)
     mul!(v̄_uy, Iv_uy, v)
    mul!(u_uz, Au_uz, u)
     mul!(w̄_uz, Iw_uz, w)

    mul!(v_vx, Av_vx, v)
     mul!(ū_vx, Iu_vx, u)
    mul!(v_vy, Av_vy, v)
     mul!(v̄_vy, Iv_vy, v)
    mul!(v_vz, Av_vz, v)
     mul!(w̄_vz, Iw_vz, w)

    mul!(w_wx, Aw_wx, w)
     mul!(ū_wx, Iu_wx, u)
    mul!(w_wy, Aw_wy, w)
     mul!(v̄_wy, Iv_wy, v)
    mul!(w_wz, Aw_wz, w)
     mul!(w̄_wz, Iw_wz, w)

      @. uū_ux = u_ux * ū_ux
      @. uv̄_uy = u_uy * v̄_uy
      @. uw̄_uz = u_uz * w̄_uz
      @. vū_vx = v_vx * ū_vx
      @. vv̄_vy = v_vy * v̄_vy
      @. vw̄_vz = v_vz * w̄_vz
      @. wū_wx = w_wx * ū_wx
      @. wv̄_wy = w_wy * v̄_wy
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

    c
end

"""
    convection_jacobian!

Compute the Jacobian of the convective forces with respect to the velocity `V`.
"""
function convection_jacobian! end

# 2D version
function convection_jacobian!(∇c, V, setup::Setup{T,2}, cache) where {T}
    (; grid, operators) = setup
    (; Cux, Cuy, Cvx, Cvy) = operators
    (; Au_ux, Au_uy, Av_vx, Av_vy) = operators
    (; Iu_ux, Iv_uy, Iu_vx, Iv_vy) = operators
    (; indu, indv) = grid
      (; ū_ux, v̄_uy) = cache
      (; ū_vx, v̄_vy) = cache
    (; Conv_ux_11, Conv_uy_11, Conv_uy_12, Conv_vx_21, Conv_vx_22, Conv_vy_22) = cache

    u = @view V[indu]
    v = @view V[indv]

    # Fill preallocated arrays
     mul!(ū_ux, Iu_ux, u)
     mul!(v̄_uy, Iv_uy, v)
     mul!(ū_vx, Iu_vx, u)
     mul!(v̄_vy, Iv_vy, v)

    ## Convective terms, u-component
     C1 = Cux * Diagonal(ū_ux)
    Conv_ux_11 .= C1 * Au_ux

     C1 = Cuy * Diagonal(v̄_uy)
    Conv_uy_11 .= C1 * Au_uy
    Conv_uy_12 .= 0

    ## Convective terms, v-component
     C1 = Cvx * Diagonal(ū_vx)
    Conv_vx_21 .= 0
    Conv_vx_22 .= C1 * Av_vx

     C1 = Cvy * Diagonal(v̄_vy)
    Conv_vy_22 .= C1 * Av_vy

    ∇c[indu, indu] = Conv_ux_11 + Conv_uy_11
    ∇c[indu, indv] = Conv_uy_12
    ∇c[indv, indu] = Conv_vx_21
    ∇c[indv, indv] = Conv_vx_22 + Conv_vy_22

    ∇c
end

# 3D version
function convection_jacobian!(∇c, V, setup::Setup{T,3}, cache) where {T}
    (; grid, operators) = setup
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = operators
    (; Au_ux, Au_uy, Au_uz) = operators
    (; Av_vx, Av_vy, Av_vz) = operators
    (; Aw_wx, Aw_wy, Aw_wz) = operators
    (; Iu_ux, Iv_uy, Iw_uz) = operators
    (; Iu_vx, Iv_vy, Iw_vz) = operators
    (; Iu_wx, Iv_wy, Iw_wz) = operators
    (; indu, indv, indw) = grid
       (; ū_ux, v̄_uy, w̄_uz) = cache
       (; ū_vx, v̄_vy, w̄_vz) = cache
       (; ū_wx, v̄_wy, w̄_wz) = cache
    (; Conv_ux_11, Conv_uy_11, Conv_uz_11, Conv_uy_12, Conv_uz_13) = cache
    (; Conv_vx_21, Conv_vx_22, Conv_vy_22, Conv_vz_22, Conv_vz_23) = cache
    (; Conv_wx_31, Conv_wy_32, Conv_wx_33, Conv_wy_33, Conv_wz_33) = cache

    u = @view V[indu]
    v = @view V[indv]
    w = @view V[indw]

    # Convection components
     mul!(ū_ux, Iu_ux, u)
     mul!(v̄_uy, Iv_uy, v)
     mul!(w̄_uz, Iw_uz, w)

     mul!(ū_vx, Iu_vx, u)
     mul!(v̄_vy, Iv_vy, v)
     mul!(w̄_vz, Iw_vz, w)

     mul!(ū_wx, Iu_wx, u)
     mul!(v̄_wy, Iv_wy, v)
     mul!(w̄_wz, Iw_wz, w)

    ## Convective terms, u-component
     C1 = Cux * Diagonal(ū_ux)
    Conv_ux_11 .= C1 * Au_ux

     C1 = Cuy * Diagonal(v̄_uy)
    Conv_uy_11 .= C1 * Au_uy
    Conv_uy_12 .= 0

     C1 = Cuz * Diagonal(w̄_uz)
    Conv_uz_11 .= C1 * Au_uz
    Conv_uz_13 .= 0

    ## Convective terms, v-component
     C1 = Cvx * Diagonal(ū_vx)
    Conv_vx_21 .= 0
    Conv_vx_22 .= C1 * Av_vx

     C1 = Cvy * Diagonal(v̄_vy)
    Conv_vy_22 .= C1 * Av_vy

     C1 = Cvz * Diagonal(w̄_vz)
    Conv_vz_23 .= 0
    Conv_vz_22 .= C1 * Av_vz

    ## Convective terms, w-component
     C1 = Cwx * Diagonal(ū_wx)
    Conv_wx_31 .= 0
    Conv_wx_33 .= C1 * Aw_wx

     C1 = Cwy * Diagonal(v̄_wy)
    Conv_wy_32 .= 0
    Conv_wy_33 .= C1 * Aw_wy

     C1 = Cwz * Diagonal(w̄_wz)
    Conv_wz_33 .= C1 * Aw_wz

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

    ∇c
end
