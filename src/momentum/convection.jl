"""
    convection(V, ϕ, t, setup, getJacobian = false)

Convenience function for initializing arrays `c` and `∇c` before filling in convection terms.
"""
function convection(V, ϕ, t, setup; getJacobian = false)
    @unpack NV = setup.grid

    cache = MomentumCache(setup)
    c = zeros(NV)
    ∇c = spzeros(NV, NV)

    convection!(c, ∇c, V, ϕ, t, setup, cache; getJacobian)
end

"""
    convection!(c, ∇c, V, ϕ, t, cache, setup, getJacobian = false) -> c, ∇c

evaluate convective terms `c` and, optionally, Jacobian `∇c = ∂c/∂V`.
"""
function convection!(c, ∇c, V, ϕ, t, setup, cache; getJacobian = false)
    @unpack order4 = setup.discretization
    @unpack regularization = setup.case
    @unpack α = setup.discretization
    @unpack indu, indv = setup.grid
    @unpack Newton_factor = setup.solver_settings
    @unpack c2, ∇c2, c3, ∇c3 = cache

    cu = @view c[indu]
    cv = @view c[indv]

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    if regularization == "no"
        # No regularization
        convection_components!(c, ∇c, V, ϕ, setup, cache; getJacobian, order4 = false)

        if order4
            convection_components!(c3, ∇c3, V, ϕ, setup, cache; getJacobian, order4)
            @. c = α * c - c3
            getJacobian && (@. ∇c = α * ∇c - ∇c3)
        end
    elseif regularization == "leray"
        # TODO: needs finishing

        # Filter the convecting field
        ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
        ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

        ϕ̄ = [ϕ̄u; ϕ̄v]

        # Divergence of filtered velocity field; should be zero!
        maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

        convection_components!(c, ∇c, V, ϕ̄, setup, cache; getJacobian)
    elseif regularization == "C2"
        ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
        ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

        ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α)
        v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α)

        ϕ̄ = [ϕ̄u; ϕ̄v]
        V̄ = [ūₕ; v̄ₕ]

        # Divergence of filtered velocity field; should be zero!
        maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

        convection_components!(c, ∇c, V̄, ϕ̄, setup, cache; getJacobian)

        cu .= filter_convection(cu, Diffu_f, yDiffu_f, α)
        cv .= filter_convection(cv, Diffv_f, yDiffv_f, α)
    elseif regularization == "C4"
        # C4 consists of 3 terms:
        # C4 = conv(filter(u), filter(u)) + filter(conv(filter(u), u') +
        #      filter(conv(u', filter(u)))
        # Where u' = u - filter(u)

        # Filter both convecting and convected velocity
        ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α)
        v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α)

        V̄ = [ūₕ; v̄ₕ]
        ΔV = V - V̄

        ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
        ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

        ϕ̄ = [ϕ̄u; ϕ̄v]
        Δϕ = ϕ - ϕ̄

        # Divergence of filtered velocity field; should be zero!
        maxdiv_f[n] = maximum(abs.(M * V̄ + yM))

        convection_components!(c, ∇c, V̄, ϕ̄, setup, cache; getJacobian)
        convection_components!(c2, ∇c2, ΔV, ϕ̄, setup, cache; getJacobian)
        convection_components!(c3, ∇c3, V̄, Δϕ, setup, cache; getJacobian)

        # TODO: consider inner loop parallelization
        # @sync begin
        #     @spawn convection_components!(c, ∇c, V̄, ϕ̄, setup, cache, getJacobian)
        #     @spawn convection_components!(c2, ∇c2, ΔV, ϕ̄, setup, cache, getJacobian)
        #     @spawn convection_components!(c3, ∇c3, V̄, Δϕ, setup, cache, getJacobian)
        # end

        cu .+= filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α)
        cv .+= filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α)
    end

    c, ∇c
end

"""
    convection_components!(c, ∇c, V, ϕ, setup, cache; getJacobian = false, order4 = false)

Compute convection components.
"""
function convection_components!(c, ∇c, V, ϕ, setup, cache; getJacobian = false, order4 = false)
    if order4
        Cux = setup.discretization.Cux3
        Cuy = setup.discretization.Cuy3
        Cvx = setup.discretization.Cvx3
        Cvy = setup.discretization.Cvy3

        Au_ux = setup.discretization.Au_ux3
        Au_uy = setup.discretization.Au_uy3
        Av_vx = setup.discretization.Av_vx3
        Av_vy = setup.discretization.Av_vy3

        yAu_ux = setup.discretization.yAu_ux3
        yAu_uy = setup.discretization.yAu_uy3
        yAv_vx = setup.discretization.yAv_vx3
        yAv_vy = setup.discretization.yAv_vy3

        Iu_ux = setup.discretization.Iu_ux3
        Iv_uy = setup.discretization.Iv_uy3
        Iu_vx = setup.discretization.Iu_vx3
        Iv_vy = setup.discretization.Iv_vy3

        yIu_ux = setup.discretization.yIu_ux3
        yIv_uy = setup.discretization.yIv_uy3
        yIu_vx = setup.discretization.yIu_vx3
        yIv_vy = setup.discretization.yIv_vy3
    else
        @unpack Cux, Cuy, Cvx, Cvy = setup.discretization
        @unpack Au_ux, Au_uy, Av_vx, Av_vy = setup.discretization
        @unpack yAu_ux, yAu_uy, yAv_vx, yAv_vy = setup.discretization
        @unpack Iu_ux, Iv_uy, Iu_vx, Iv_vy = setup.discretization
        @unpack yIu_ux, yIv_uy, yIu_vx, yIv_vy = setup.discretization
    end

    @unpack indu, indv = setup.grid
    @unpack Newton_factor = setup.solver_settings

    @unpack u_ux, ū_ux, uū_ux, u_uy, v̄_uy, uv̄_uy = cache
    @unpack v_vx, ū_vx, vū_vx, v_vy, v̄_vy, vv̄_vy = cache
    @unpack ∂uū∂x, ∂uv̄∂y, ∂vū∂x, ∂vv̄∂y = cache
    @unpack Conv_ux_11, Conv_uy_11, Conv_uy_12, Conv_vx_21, Conv_vx_22, Conv_vy_22 = cache

    cu = @view c[indu]
    cv = @view c[indv]

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # Convection components
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

    mul!(∂uū∂x, Cux, uū_ux)
    mul!(∂uv̄∂y, Cuy, uv̄_uy)
    mul!(∂vū∂x, Cvx, vū_vx)
    mul!(∂vv̄∂y, Cvy, vv̄_vy)

    # U_ux = Au_ux * uₕ + yAu_ux                # U at ux
    # Ū_ux = Iu_ux * ϕu + yIu_ux                # Ū at ux
    # ∂uū∂x = Cux * (u_ux .* ū_ux)

    # U_uy = Au_uy * uₕ + yAu_uy                # U at uy
    # V̄_uy = Iv_uy * ϕv + yIv_uy                # Ū at uy
    # ∂uv̄∂y = Cuy * (u_uy .* v̄_uy)

    # V_vx = Av_vx * vₕ + yAv_vx                # V at vx
    # Ū_vx = Iu_vx * ϕu + yIu_vx                # Ū at vx
    # ∂vū∂x = Cvx * (v_vx .* ū_vx)

    # V_vy = Av_vy * vₕ + yAv_vy                # V at vy
    # V̄_vy = Iv_vy * ϕv + yIv_vy                # Ū at vy
    # ∂vv̄∂y = Cvy * (v_vy .* v̄_vy)

    @. cu = ∂uū∂x + ∂uv̄∂y
    @. cv = ∂vū∂x + ∂vv̄∂y

    if getJacobian
        Jux = @view ∇c[indu, indu]
        Juy = @view ∇c[indu, indv]
        Jvx = @view ∇c[indv, indu]
        Jvy = @view ∇c[indv, indv]

        ## Convective terms, u-component
        C1 = Cux * Diagonal(ū_ux)
        C2 = Cux * Diagonal(u_ux) * Newton_factor
        Conv_ux_11 .= C1 * Au_ux .+ C2 * Iu_ux
        # mul!(Conv_ux_11, C1, Au_ux)
        # mul!(Conv_ux_11, C2, Iu_ux, 1, 1)

        C1 = Cuy * Diagonal(v̄_uy)
        C2 = Cuy * Diagonal(u_uy) * Newton_factor
        # mul!(Conv_uy_11, C1, Au_uy)
        # mul!(Conv_uy_12, C2, Iv_uy)
        Conv_uy_11 .= C1 * Au_uy
        Conv_uy_12 .= C2 * Iv_uy

        @. Jux = Conv_ux_11 + Conv_uy_11
        @. Juy = Conv_uy_12

        ## Convective terms, v-component
        C1 = Cvx * Diagonal(ū_vx)
        C2 = Cvx * Diagonal(v_vx) * Newton_factor
        # mul!(Conv_vx_21, C2, Iu_vx)
        # mul!(Conv_vx_22, C1, Av_vx)
        Conv_vx_21 .= C2 * Iu_vx
        Conv_vx_22 .= C1 * Av_vx

        C1 = Cvy * Diagonal(v̄_vy)
        C2 = Cvy * Diagonal(v_vy) * Newton_factor
        Conv_vy_22 .= C1 * Av_vy .+ C2 * Iv_vy
        # mul!(Conv_vy_22, C1, Av_vy)
        # mul!(Conv_vy_22, C2, Iv_vy, 1, 1)

        @. Jvx = Conv_vx_21
        @. Jvy = Conv_vx_22 + Conv_vy_22
    end

    c, ∇c
end
