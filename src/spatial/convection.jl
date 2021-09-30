function convection(V, ϕ, t, setup, getJacobian)
    @unpack NV = setup.grid

    cache = MomentumCache(setup)
    c = zeros(NV)
    ∇c = spzeros(NV, NV)

    convection!(c, ∇c, V, ϕ, t, setup, cache, getJacobian)
end


"""
    convection!(c, ∇c, V, ϕ, t, cache, setup, getJacobian) -> c, ∇c

evaluate convective terms and, optionally, Jacobians
V: velocity field
ϕ: "convection" field: e.g. d(ϕ_x u)/dx + d(ϕ_y u)/dy; usually ϕ_x = u, ϕ_y = v
"""
function convection!(c, ∇c, V, ϕ, t, setup, cache, getJacobian)
    @unpack order4 = setup.discretization
    @unpack regularization = setup.case
    @unpack α = setup.discretization
    @unpack Nu, Nv, NV, indu, indv = setup.grid
    @unpack Newton_factor = setup.solver_settings
    @unpack c2, ∇c2, c3, ∇c3 = cache

    cu = @view c[indu]
    cv = @view c[indv]

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    if regularization == "no"
        # no regularization
        convection_components!(c, ∇c, V, ϕ, setup, getJacobian, false)

        if order4
            convection_components!(c3, ∇c3, V, ϕ, setup, getJacobian, true)
            @. c = α * c - c3
            getJacobian && (@. ∇c = α * ∇c - ∇c3)
        end
    elseif regularization == "leray"
        # TODO: needs finishing

        # filter the convecting field
        ϕu_f = filter_convection(ϕu, Diffu_f, yDiffu_f, α) #uₕ + (α^2)*Re*(Diffu*uₕ + yDiffu);
        ϕv_f = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

        ϕ_filtered = [ϕu_f; ϕv_f]

        # divergence of filtered velocity field; should be zero!
        maxdiv_f = maximum(abs.(M * ϕ_filtered + yM))

        convection_components!(c, ∇c, V, ϕ_filtered, setup, getJacobian)
    elseif regularization == "C2"
        ϕu_f = filter_convection(ϕu, Diffu_f, yDiffu_f, α) #uₕ + (α^2)*Re*(Diffu*uₕ + yDiffu);
        ϕv_f = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

        uₕ_f = filter_convection(uₕ, Diffu_f, yDiffu_f, α) #uₕ + (α^2)*Re*(Diffu*uₕ + yDiffu);
        vₕ_f = filter_convection(vₕ, Diffv_f, yDiffv_f, α)

        ϕ_filtered = [ϕu_f; ϕv_f]
        V_filtered = [uₕ_f; vₕ_f]

        # divergence of filtered velocity field; should be zero!
        maxdiv_f = maximum(abs.(M * ϕ_filtered + yM))

        convection_components!(c, ∇c, V_filtered, ϕ_filtered, setup, getJacobian)

        cu .= filter_convection(cu, Diffu_f, yDiffu_f, α)
        cv .= filter_convection(cv, Diffv_f, yDiffv_f, α)
    elseif regularization == "C4"
        # C4 consists of 3 terms:
        # C4 = conv(filter(u), filter(u)) + filter(conv(filter(u), u') +
        #      filter(conv(u', filter(u)))
        # where u' = u - filter(u)

        # filter both convecting and convected velocity
        uₕ_f = filter_convection(uₕ, Diffu_f, yDiffu_f, α) #uₕ + (α^2)*Re*(Diffu*uₕ + yDiffu);
        vₕ_f = filter_convection(vₕ, Diffv_f, yDiffv_f, α)

        V_filtered = [uₕ_f; vₕ_f]

        dV = V - V_filtered

        ϕu_f = filter_convection(ϕu, Diffu_f, yDiffu_f, α) #uₕ + (α^2)*Re*(Diffu*uₕ + yDiffu);
        ϕv_f = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

        ϕ_filtered = [ϕu_f; ϕv_f]
        Δϕ = ϕ - ϕ_filtered

        # divergence of filtered velocity field; should be zero!
        maxdiv_f[n] = maximum(abs.(M * V_filtered + yM))

        convection_components!(c, ∇c, V_filtered, ϕ_filtered, setup, getJacobian)
        convection_components!(c2, ∇c2, dV, ϕ_filtered, setup, getJacobian)
        convection_components!(c3, ∇c3, V_filtered, Δϕ, setup, getJacobian)

        cu .+= filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α)
        cv .+= filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α)
    end

    c, ∇c
end

function convection_components!(c, ∇c, V, ϕ, setup, getJacobian, order4 = false)
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

    @unpack Nu, Nv, NV, indu, indv = setup.grid

    cu = @view c[indu]
    cv = @view c[indv]

    if getJacobian
        Jacu = @view ∇c[indu, :]
        Jacv = @view ∇c[indv, :]
    end

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    u_ux = Au_ux * uₕ + yAu_ux                 # u at ux
    uf_ux = Iu_ux * ϕu + yIu_ux                # ubar at ux
    du2dx = Cux * (uf_ux .* u_ux)

    u_uy = Au_uy * uₕ + yAu_uy                 # u at uy
    vf_uy = Iv_uy * ϕv + yIv_uy                # vbar at uy
    duvdy = Cuy * (vf_uy .* u_uy)

    v_vx = Av_vx * vₕ + yAv_vx                 # v at vx
    uf_vx = Iu_vx * ϕu + yIu_vx                # ubar at vx
    duvdx = Cvx * (uf_vx .* v_vx)

    v_vy = Av_vy * vₕ + yAv_vy                 # v at vy
    vf_vy = Iv_vy * ϕv + yIv_vy                # vbar at vy
    dv2dy = Cvy * (vf_vy .* v_vy)

    @. cu = du2dx + duvdy
    @. cv = duvdx + dv2dy

    if getJacobian
        ## convective terms, u-component
        # c^n * u^(n+1), c = u
        C1 = Cux * spdiagm(uf_ux)
        C2 = Cux * spdiagm(u_ux) * Newton_factor
        Conv_ux_11 = C1 * Au_ux + C2 * Iu_ux

        C1 = Cuy * spdiagm(vf_uy)
        C2 = Cuy * spdiagm(u_uy) * Newton_factor
        Conv_uy_11 = C1 * Au_uy
        Conv_uy_12 = C2 * Iv_uy

        Jacu .= [(Conv_ux_11 + Conv_uy_11) Conv_uy_12]

        ## convective terms, v-component
        C1 = Cvx * spdiagm(uf_vx)
        C2 = Cvx * spdiagm(v_vx) * Newton_factor
        Conv_vx_21 = C2 * Iu_vx
        Conv_vx_22 = C1 * Av_vx

        C1 = Cvy * spdiagm(vf_vy)
        C2 = Cvy * spdiagm(v_vy) * Newton_factor
        Conv_vy_22 = C1 * Av_vy + C2 * Iv_vy

        Jacv .= [Conv_vx_21 (Conv_vx_22 + Conv_vy_22)]
    end

    c, ∇c
end
