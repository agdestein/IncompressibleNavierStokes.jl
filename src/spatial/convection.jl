"""
    convection(V, C, t, setup, getJacobian)

evaluate convective terms and, optionally, Jacobians
V: velocity field
C: 'convection' field: e.g. d(c_x u)/dx + d(c_y u)/dy; usually c_x = u,
c_y = v
"""
function convection(V, C, t, options, getJacobian)
    # evaluate convective terms and, optionally, Jacobians
    # V: velocity field
    # C: 'convection' field: e.g. d(c_x u)/dx + d(c_y u)/dy; usually c_x = u,
    # c_y = v

    α = options.discretization.α
    order4 = options.discretization.order4

    regularize = options.case.regularize

    @unpack Nu, Nv, NV, indu, indv = setup.grid

    Jacu = sparse(Nu, NV)
    Jacv = sparse(Nv, NV)

    uh = V[indu]
    vh = V[indv]

    cu = C[indu]
    cv = C[indv]

    if regularize == 0
        # no regularization
        convu, convv, Jacu, Jacv = convection_components(C, V, options, getJacobian, false)

        if order4 == 1
            convu3, convv3, Jacu3, Jacv3 =
                convection_components(C, V, options, getJacobian, true)
            convu = α * convu - convu3
            convv = α * convv - convv3
            Jacu = α * Jacu - Jacu3
            Jacv = α * Jacv - Jacv3
        end
    elseif regularize == 1
        # Leray
        # TODO: needs finishing

        # filter the convecting field
        cu_f = filter_convection(cu, Diffu_f, yDiffu_f, α) #uh + (α^2)*Re*(Diffu*uh + yDiffu);
        cv_f = filter_convection(cv, Diffv_f, yDiffv_f, α)

        C_filtered = [cu_f; cv_f]

        # divergence of filtered velocity field; should be zero!
        maxdiv_f = max(abs(M * C_filtered + yM))

        convu, convv, Jacu, Jacv =
            convection_components(C_filtered, V, options, getJacobian)
    elseif regularize == 2
        ## C2

        cu_f = filter_convection(cu, Diffu_f, yDiffu_f, α) #uh + (α^2)*Re*(Diffu*uh + yDiffu);
        cv_f = filter_convection(cv, Diffv_f, yDiffv_f, α)

        uh_f = filter_convection(uh, Diffu_f, yDiffu_f, α) #uh + (α^2)*Re*(Diffu*uh + yDiffu);
        vh_f = filter_convection(vh, Diffv_f, yDiffv_f, α)

        C_filtered = [cu_f; cv_f]
        V_filtered = [uh_f; vh_f]

        # divergence of filtered velocity field; should be zero!
        maxdiv_f = max(abs(M * C_filtered + yM))

        [convu, convv, Jacu, Jacv] =
            convection_components(C_filtered, V_filtered, options, getJacobian)

        convu = filter_convection(convu, Diffu_f, yDiffu_f, α)
        convv = filter_convection(convv, Diffv_f, yDiffv_f, α)
    elseif regularize == 4
        # C4 consists of 3 terms:
        # C4 = conv(filter(u), filter(u)) + filter(conv(filter(u), u') +
        #      filter(conv(u', filter(u)))
        # where u' = u - filter(u)

        # filter both convecting and convected velocity
        uh_f = filter_convection(uh, Diffu_f, yDiffu_f, α) #uh + (α^2)*Re*(Diffu*uh + yDiffu);
        vh_f = filter_convection(vh, Diffv_f, yDiffv_f, α)

        V_filtered = [uh_f; vh_f]

        dV = V - V_filtered

        cu_f = filter_convection(cu, Diffu_f, yDiffu_f, α) #uh + (α^2)*Re*(Diffu*uh + yDiffu);
        cv_f = filter_convection(cv, Diffv_f, yDiffv_f, α)

        C_filtered = [cu_f; cv_f]
        dC = C - C_filtered


        # divergence of filtered velocity field; should be zero!
        maxdiv_f[n] = max(abs(M * V_filtered + yM))

        convu1, convv1, Jacu, Jacv =
            convection_components(C_filtered, V_filtered, options, getJacobian)

        convu2, convv2, Jacu, Jacv =
            convection_components(C_filtered, dV, options, getJacobian)

        convu3, convv3, Jacu, Jacv =
            convection_components(dC, V_filtered, options, getJacobian)

        convu = convu1 + filter_convection(convu2 + convu3, Diffu_f, yDiffu_f, α)
        convv = convv1 + filter_convection(convv2 + convv3, Diffv_f, yDiffv_f, α)
    end
    convu, convv, Jacu, Jacv
end

function convection_components(C, V, options, getJacobian, order4 = false)

    if order4
        Cux = options.discretization.Cux3
        Cuy = options.discretization.Cuy3
        Cvx = options.discretization.Cvx3
        Cvy = options.discretization.Cvy3

        Au_ux = options.discretization.Au_ux3
        Au_uy = options.discretization.Au_uy3
        Av_vx = options.discretization.Av_vx3
        Av_vy = options.discretization.Av_vy3

        yAu_ux = options.discretization.yAu_ux3
        yAu_uy = options.discretization.yAu_uy3
        yAv_vx = options.discretization.yAv_vx3
        yAv_vy = options.discretization.yAv_vy3

        Iu_ux = options.discretization.Iu_ux3
        Iv_uy = options.discretization.Iv_uy3
        Iu_vx = options.discretization.Iu_vx3
        Iv_vy = options.discretization.Iv_vy3

        yIu_ux = options.discretization.yIu_ux3
        yIv_uy = options.discretization.yIv_uy3
        yIu_vx = options.discretization.yIu_vx3
        yIv_vy = options.discretization.yIv_vy3
    else
        Cux = options.discretization.Cux
        Cuy = options.discretization.Cuy
        Cvx = options.discretization.Cvx
        Cvy = options.discretization.Cvy

        Au_ux = options.discretization.Au_ux
        Au_uy = options.discretization.Au_uy
        Av_vx = options.discretization.Av_vx
        Av_vy = options.discretization.Av_vy

        yAu_ux = options.discretization.yAu_ux
        yAu_uy = options.discretization.yAu_uy
        yAv_vx = options.discretization.yAv_vx
        yAv_vy = options.discretization.yAv_vy

        Iu_ux = options.discretization.Iu_ux
        Iv_uy = options.discretization.Iv_uy
        Iu_vx = options.discretization.Iu_vx
        Iv_vy = options.discretization.Iv_vy

        yIu_ux = options.discretization.yIu_ux
        yIv_uy = options.discretization.yIv_uy
        yIu_vx = options.discretization.yIu_vx
        yIv_vy = options.discretization.yIv_vy
    end

    indu = options.grid.indu
    indv = options.grid.indv

    Nu = options.grid.Nu
    Nv = options.grid.Nv
    NV = options.grid.NV

    Jacu = sparse(Nu, NV)
    Jacv = sparse(Nv, NV)

    uh = V[indu]
    vh = V[indv]

    cu = C[indu]
    cv = C[indv]

    u_ux = Au_ux * uh + yAu_ux                 # u at ux
    uf_ux = Iu_ux * cu + yIu_ux                 # ubar at ux
    du2dx = Cux * (uf_ux .* u_ux)

    u_uy = Au_uy * uh + yAu_uy                 # u at uy
    vf_uy = Iv_uy * cv + yIv_uy                 # vbar at uy
    duvdy = Cuy * (vf_uy .* u_uy)

    v_vx = Av_vx * vh + yAv_vx                 # v at vx
    uf_vx = Iu_vx * cu + yIu_vx                 # ubar at vx
    duvdx = Cvx * (uf_vx .* v_vx)

    v_vy = Av_vy * vh + yAv_vy                 # v at vy
    vf_vy = Iv_vy * cv + yIv_vy                 # vbar at vy
    dv2dy = Cvy * (vf_vy .* v_vy)

    convu = du2dx + duvdy
    convv = duvdx + dv2dy

    if getJacobian
        Newton = options.solversettings.Newton_factor
        N1 = length(u_ux) #options.grid.N1;
        N2 = length(u_uy) #options.grid.N2;
        N3 = length(v_vx) #options.grid.N3;
        N4 = length(v_vy) #options.grid.N4;

        ## convective terms, u-component
        # c^n * u^(n+1), c = u
        C1 = Cux * spdiagm(uf_ux)
        C2 = Cux * spdiagm(u_ux) * Newton
        Conv_ux_11 = C1 * Au_ux + C2 * Iu_ux

        C1 = Cuy * spdiagm(vf_uy)
        C2 = Cuy * spdiagm(u_uy) * Newton
        Conv_uy_11 = C1 * Au_uy
        Conv_uy_12 = C2 * Iv_uy

        Jacu = [Conv_ux_11 + Conv_uy_11 Conv_uy_12]

        ## convective terms, v-component
        C1 = Cvx * spdiagm(uf_vx)
        C2 = Cvx * spdiagm(v_vx) * Newton
        Conv_vx_21 = C2 * Iu_vx
        Conv_vx_22 = C1 * Av_vx

        C1 = Cvy * spdiagm(vf_vy)
        C2 = Cvy * spdiagm(v_vy) * Newton
        Conv_vy_22 = C1 * Av_vy + C2 * Iv_vy

        Jacv = [Conv_vx_21 Conv_vx_22 + Conv_vy_22]
    end

    convu, convv, Jacu, Jacv
end
