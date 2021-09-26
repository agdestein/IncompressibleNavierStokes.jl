"""
    set_bc_vectors!(setup, t)

Construct boundary conditions
"""
function set_bc_vectors!(setup, t)

    # steady
    steady = setup.case.steady

    # 4th order
    order4 = setup.discretization.order4

    # Reynolds number
    Re = setup.fluid.Re

    # boundary conditions
    BC = setup.BC

    global uBC, vBC, dudtBC, dvdtBC

    # type of stress tensor
    visc = setup.case.visc

    if order4
        alfa = setup.discretization.alfa
    end

    ## grid settings

    # number of interior points and boundary points
    @unpack Nux_in, Nvy_in, Np, Npx, Npy, xin, yin, x, y, hx, hy, xp, yp = setup.grid

    ## get BC values
    uLo = uBC(x, y[1], t, setup)
    uUp = uBC(x, y[end], t, setup)
    # uLe = uBC(x[1], y, t, setup);
    # uRi = uBC(x[end], y, t, setup);

    uLo_i = uBC(xin, y[1], t, setup)
    uUp_i = uBC(xin, y[end], t, setup)
    uLe_i = uBC(x[1], yp, t, setup)
    uRi_i = uBC(x[end], yp, t, setup)

    # vLo = vBC(x, y[1], t, setup);
    # vUp = vBC(x, y[end], t, setup);
    vLe = vBC(x[1], y, t, setup)
    vRi = vBC(x[end], y, t, setup)

    vLo_i = vBC(xp, y[1], t, setup)
    vUp_i = vBC(xp, y[end], t, setup)
    vLe_i = vBC(x[1], yin, t, setup)
    vRi_i = vBC(x[end], yin, t, setup)

    if !steady && setup.BC.BC_unsteady
        dudtLe_i = dudtBC(x[1], setup.grid.yp, t, setup)
        dudtRi_i = dudtBC(x[end], setup.grid.yp, t, setup)

        dvdtLo_i = dvdtBC(setup.grid.xp, y[1], t, setup)
        dvdtUp_i = dvdtBC(setup.grid.xp, y[end], t, setup)
    end

    @unpack pLe, pRi, pLo, pUp = setup.bc

    ## boundary conditions for divergence
    Mx_BC = setup.discretization.Mx_BC
    My_BC = setup.discretization.My_BC

    if order4
        Mx_BC3 = setup.discretization.Mx_BC3
        My_BC3 = setup.discretization.My_BC3
    end

    # Mx
    ybc = kron(uLe_i, Mx_BC.ybc1) + kron(uRi_i, Mx_BC.ybc2)
    yMx = Mx_BC.Bbc * ybc
    if order4
        ybc3 = kron(uLe_i, Mx_BC3.ybc1) + kron(uRi_i, Mx_BC3.ybc2)
        yMx3 = Mx_BC3.Bbc * ybc3
        yMx = alfa * yMx - yMx3
    end

    # My
    ybc = kron(My_BC.ybc1, vLo_i) + kron(My_BC.ybc2, vUp_i)
    yMy = My_BC.Bbc * ybc
    if order4
        ybc3 = kron(My_BC3.ybc1, vLo_i) + kron(My_BC3.ybc2, vUp_i)
        yMy3 = My_BC3.Bbc * ybc3
        yMy = alfa * yMy - yMy3
    end

    yM = yMx + yMy
    setup.discretization.yM = yM


    ## time derivative of divergence
    if ~steady
        if setup.BC.BC_unsteady
            ybc = kron(dudtLe_i, Mx_BC.ybc1) + kron(dudtRi_i, Mx_BC.ybc2)
            ydMx = Mx_BC.Bbc * ybc
            if order4
                ybc3 = kron(dudtLe_i, Mx_BC3.ybc1) + kron(dudtRi_i, Mx_BC3.ybc2)
                ydMx3 = Mx_BC3.Bbc * ybc3
                ydMx = alfa * ydMx - ydMx3
            end

            # My
            ybc = kron(My_BC.ybc1, dvdtLo_i) + kron(My_BC.ybc2, dvdtUp_i)
            ydMy = My_BC.Bbc * ybc
            if order4
                ybc3 = kron(My_BC3.ybc1, dvdtLo_i) + kron(My_BC3.ybc2, dvdtUp_i)
                ydMy3 = My_BC3.Bbc * ybc3
                ydMy = alfa * ydMy - ydMy3
            end

            ydM = ydMx + ydMy

            setup.discretization.ydM = ydM
        else
            setup.discretization.ydM = zeros(Np, 1)
        end
    end

    # if ibm == 1
    #     ydM = [ydM; zeros(n_ibm, 1)];
    #     yM = [yM; zeros(n_ibm, 1)];
    # end

    ## boundary conditions for pressure

    # left and right side
    y1D_le = zeros(Nux_in, 1)
    y1D_ri = zeros(Nux_in, 1)
    if BC.u.left == "pres"
        y1D_le[1] = -1
    end
    if BC.u.right == "pres"
        y1D_ri[end] = 1
    end
    y_px = kron(hy .* pLe, y1D_le) + kron(hy .* pRi, y1D_ri)

    # lower and upper side
    y1D_lo = zeros(Nvy_in, 1)
    y1D_up = zeros(Nvy_in, 1)
    if strcmp(BC.v.low, "pres")
        y1D_lo[1] = -1
    end
    if strcmp(BC.v.up, "pres")
        y1D_up[end] = 1
    end
    y_py = kron(y1D_lo, hx .* pLo) + kron(y1D_up, hx .* pUp)


    setup.discretization.y_px = y_px
    setup.discretization.y_py = y_py


    ## boundary conditions for averaging

    Au_ux_BC = setup.discretization.Au_ux_BC
    Au_uy_BC = setup.discretization.Au_uy_BC
    Av_vx_BC = setup.discretization.Av_vx_BC
    Av_vy_BC = setup.discretization.Av_vy_BC
    if order4
        Au_ux_BC3 = setup.discretization.Au_ux_BC3
        Au_uy_BC3 = setup.discretization.Au_uy_BC3
        Av_vx_BC3 = setup.discretization.Av_vx_BC3
        Av_vy_BC3 = setup.discretization.Av_vy_BC3
    end

    # Au_ux
    # uLe_i = interp1(y, uLe, yp);
    # uRi_i = interp1(y, uRi, yp);
    ybc = kron(uLe_i, Au_ux_BC.ybc1) + kron(uRi_i, Au_ux_BC.ybc2)
    yAu_ux = Au_ux_BC.Bbc * ybc
    if order4
        ybc3 = kron(uLe_i, Au_ux_BC3.ybc1) + kron(uRi_i, Au_ux_BC3.ybc2)
        yAu_ux3 = Au_ux_BC3.Bbc * ybc3
    end

    # Au_uy
    ybc = kron(Au_uy_BC.ybc1, uLo_i) + kron(Au_uy_BC.ybc2, uUp_i)
    yAu_uy = Au_uy_BC.Bbc * ybc
    if order4
        ybc3 = kron(Au_uy_BC3.ybc1, uLo_i) + kron(Au_uy_BC3.ybc2, uUp_i)
        yAu_uy3 = Au_uy_BC3.Bbc * ybc3
    end

    # Av_vx
    ybc = kron(vLe_i, Av_vx_BC.ybc1) + kron(vRi_i, Av_vx_BC.ybc2)
    yAv_vx = Av_vx_BC.Bbc * ybc
    if order4
        ybc3 = kron(vLe_i, Av_vx_BC3.ybc1) + kron(vRi_i, Av_vx_BC3.ybc2)
        yAv_vx3 = Av_vx_BC3.Bbc * ybc3
    end

    # Av_vy
    # vLo_i = interp1(x, vLo, xp);
    # vUp_i = interp1(x, vUp, xp);
    ybc = kron(Av_vy_BC.ybc1, vLo_i) + kron(Av_vy_BC.ybc2, vUp_i)
    yAv_vy = Av_vy_BC.Bbc * ybc
    if order4
        ybc3 = kron(Av_vy_BC3.ybc1, vLo_i) + kron(Av_vy_BC3.ybc2, vUp_i)
        yAv_vy3 = Av_vy_BC3.Bbc * ybc3
    end

    setup.discretization.yAu_ux = yAu_ux
    setup.discretization.yAu_uy = yAu_uy
    setup.discretization.yAv_vx = yAv_vx
    setup.discretization.yAv_vy = yAv_vy

    if order4
        setup.discretization.yAu_ux3 = yAu_ux3
        setup.discretization.yAu_uy3 = yAu_uy3
        setup.discretization.yAv_vx3 = yAv_vx3
        setup.discretization.yAv_vy3 = yAv_vy3
    end

    ## boundary conditions for diffusion

    Su_ux_BC = setup.discretization.Su_ux_BC
    Su_uy_BC = setup.discretization.Su_uy_BC
    Sv_vx_BC = setup.discretization.Sv_vx_BC
    Sv_vy_BC = setup.discretization.Sv_vy_BC
    Dux = setup.discretization.Dux
    Duy = setup.discretization.Duy
    Dvx = setup.discretization.Dvx
    Dvy = setup.discretization.Dvy


    if order4 == 0

        Su_vx_BC_lr = setup.discretization.Su_vx_BC_lr
        Su_vx_BC_lu = setup.discretization.Su_vx_BC_lu
        Sv_uy_BC_lr = setup.discretization.Sv_uy_BC_lr
        Sv_uy_BC_lu = setup.discretization.Sv_uy_BC_lu

        # Su_ux
        # uLe_i = interp1(y, uLe, yp);
        # uRi_i = interp1(y, uRi, yp);
        ybc = kron(uLe_i, Su_ux_BC.ybc1) + kron(uRi_i, Su_ux_BC.ybc2)
        ySu_ux = Su_ux_BC.Bbc * ybc

        # Su_uy
        # uLo_i = interp1(x, uLo, xin);
        # uUp_i = interp1(x, uUp, xin);
        ybc = kron(Su_uy_BC.ybc1, uLo_i) + kron(Su_uy_BC.ybc2, uUp_i)
        ySu_uy = Su_uy_BC.Bbc * ybc

        Sv_uy = Sv_uy_BC_lr.B2D * Sv_uy_BC_lu.B2D

        # Sv_uy (left/right)
        ybc = kron(vLe, Sv_uy_BC_lr.ybc1) + kron(vRi, Sv_uy_BC_lr.ybc2)
        ySv_uy_lr = Sv_uy_BC_lr.Bbc * ybc
        # Iv_uy (low/up)
        # vLo_i = interp1(x, vLo, xp);
        # vUp_i = interp1(x, vUp, xp);
        ybc = kron(Sv_uy_BC_lu.ybc1, vLo_i) + kron(Sv_uy_BC_lu.ybc2, vUp_i)
        ySv_uy_lu = Sv_uy_BC_lr.B2D * Sv_uy_BC_lu.Bbc * ybc

        ySv_uy = ySv_uy_lr + ySv_uy_lu

        # Su_vx (low/up)
        ybc = kron(Su_vx_BC_lu.ybc1, uLo) + kron(Su_vx_BC_lu.ybc2, uUp)
        ySu_vx_lu = Su_vx_BC_lu.Bbc * ybc
        # Su_vx (left/right)
        # uLe_i = interp1(y, uLe, yp);
        # uRi_i = interp1(y, uRi, yp);
        ybc = kron(uLe_i, Su_vx_BC_lr.ybc1) + kron(uRi_i, Su_vx_BC_lr.ybc2)
        ySu_vx_lr = Su_vx_BC_lu.B2D * Su_vx_BC_lr.Bbc * ybc
        ySu_vx = ySu_vx_lr + ySu_vx_lu

        # Sv_vx
        # vLe_i = interp1(y, vLe, yin);
        # vRi_i = interp1(y, vRi, yin);
        ybc = kron(vLe_i, Sv_vx_BC.ybc1) + kron(vRi_i, Sv_vx_BC.ybc2)
        ySv_vx = Sv_vx_BC.Bbc * ybc

        # Sv_vy
        # vLo_i = interp1(x, vLo, xp);
        # vUp_i = interp1(x, vUp, xp);
        ybc = kron(Sv_vy_BC.ybc1, vLo_i) + kron(Sv_vy_BC.ybc2, vUp_i)
        ySv_vy = Sv_vy_BC.Bbc * ybc

        if visc == "laminar"
            yDiffu = Dux * ((1 / Re) * ySu_ux) + Duy * ((1 / Re) * ySu_uy)
            yDiffv = Dvx * ((1 / Re) * ySv_vx) + Dvy * ((1 / Re) * ySv_vy)

            setup.discretization.yDiffu = yDiffu
            setup.discretization.yDiffv = yDiffv
        elseif visc ∈ ["keps", "LES", "qr", "ML"]
            # instead, we will use the following values directly (see
            # diffusion.m and strain_tensor.m)
            setup.discretization.ySu_ux = ySu_ux
            setup.discretization.ySu_uy = ySu_uy
            setup.discretization.ySu_vx = ySu_vx
            setup.discretization.ySv_vx = ySv_vx
            setup.discretization.ySv_vy = ySv_vy
            setup.discretization.ySv_uy = ySv_uy
        end
    end

    if order4

        Su_ux_BC3 = setup.discretization.Su_ux_BC3
        Su_uy_BC3 = setup.discretization.Su_uy_BC3
        Sv_vx_BC3 = setup.discretization.Sv_vx_BC3
        Sv_vy_BC3 = setup.discretization.Sv_vy_BC3
        Diffux_div = setup.discretization.Diffux_div
        Diffuy_div = setup.discretization.Diffuy_div
        Diffvx_div = setup.discretization.Diffvx_div
        Diffvy_div = setup.discretization.Diffvy_div

        ybc1 = kron(uLe_i, Su_ux_BC.ybc1) + kron(uRi_i, Su_ux_BC.ybc2)
        ybc3 = kron(uLe_i, Su_ux_BC3.ybc1) + kron(uRi_i, Su_ux_BC3.ybc2)
        ySu_ux = alfa * Su_ux_BC.Bbc * ybc1 - Su_ux_BC3.Bbc * ybc3

        ybc1 = kron(Su_uy_BC.ybc1, uLo_i) + kron(Su_uy_BC.ybc2, uUp_i)
        ybc3 = kron(Su_uy_BC3.ybc1, uLo_i) + kron(Su_uy_BC3.ybc2, uUp_i)
        ySu_uy = alfa * Su_uy_BC.Bbc * ybc1 - Su_uy_BC3.Bbc * ybc3

        ybc1 = kron(vLe_i, Sv_vx_BC.ybc1) + kron(vRi_i, Sv_vx_BC.ybc2)
        ybc3 = kron(vLe_i, Sv_vx_BC3.ybc1) + kron(vRi_i, Sv_vx_BC3.ybc2)
        ySv_vx = alfa * Sv_vx_BC.Bbc * ybc1 - Sv_vx_BC3.Bbc * ybc3

        ybc1 = kron(Sv_vy_BC.ybc1, vLo_i) + kron(Sv_vy_BC.ybc2, vUp_i)
        ybc3 = kron(Sv_vy_BC3.ybc1, vLo_i) + kron(Sv_vy_BC3.ybc2, vUp_i)
        ySv_vy = alfa * Sv_vy_BC.Bbc * ybc1 - Sv_vy_BC3.Bbc * ybc3

        if visc == "laminar"
            yDiffu = (1 / Re) * (Diffux_div * ySu_ux + Diffuy_div * ySu_uy)
            yDiffv = (1 / Re) * (Diffvx_div * ySv_vx + Diffvy_div * ySv_vy)

            setup.discretization.yDiffu = yDiffu
            setup.discretization.yDiffv = yDiffv
        elseif visc ∈ ["keps", "LES", "qr", "ML"]
            error("fourth order turbulent diffusion not implemented")
        end
    end

    ## boundary conditions for interpolation
    Iu_ux_BC = setup.discretization.Iu_ux_BC
    Iv_uy_BC_lr = setup.discretization.Iv_uy_BC_lr
    Iv_uy_BC_lu = setup.discretization.Iv_uy_BC_lu
    Iu_vx_BC_lr = setup.discretization.Iu_vx_BC_lr
    Iu_vx_BC_lu = setup.discretization.Iu_vx_BC_lu
    Iv_vy_BC = setup.discretization.Iv_vy_BC

    if order4
        Iu_ux_BC3 = setup.discretization.Iu_ux_BC3
        Iv_uy_BC_lu3 = setup.discretization.Iv_uy_BC_lu3
        Iv_uy_BC_lr3 = setup.discretization.Iv_uy_BC_lr3
        Iu_vx_BC_lu3 = setup.discretization.Iu_vx_BC_lu3
        Iu_vx_BC_lr3 = setup.discretization.Iu_vx_BC_lr3
        Iv_vy_BC3 = setup.discretization.Iv_vy_BC3
    end

    # Iu_ux
    # uLe_i = interp1(y, uLe, yp);
    # uRi_i = interp1(y, uRi, yp);
    ybc = kron(uLe_i, Iu_ux_BC.ybc1) + kron(uRi_i, Iu_ux_BC.ybc2)
    yIu_ux = Iu_ux_BC.Bbc * ybc
    if order4
        ybc3 = kron(uLe_i, Iu_ux_BC3.ybc1) + kron(uRi_i, Iu_ux_BC3.ybc2)
        yIu_ux3 = Iu_ux_BC3.Bbc * ybc3
    end

    # Iv_uy (left/right)
    ybc = kron(vLe, Iv_uy_BC_lr.ybc1) + kron(vRi, Iv_uy_BC_lr.ybc2)
    yIv_uy_lr = Iv_uy_BC_lr.Bbc * ybc
    # Iv_uy (low/up)
    # vLo_i = interp1(x, vLo, xp);
    # vUp_i = interp1(x, vUp, xp);
    ybc = kron(Iv_uy_BC_lu.ybc1, vLo_i) + kron(Iv_uy_BC_lu.ybc2, vUp_i)
    yIv_uy_lu = Iv_uy_BC_lr.B2D * Iv_uy_BC_lu.Bbc * ybc
    yIv_uy = yIv_uy_lr + yIv_uy_lu

    if order4
        if BC.v.low == "dir"
            vLe_ext = [2 * vLe[1] - vLe(2); vLe]
            vRi_ext = [2 * vRi[1] - vRi(2); vRi]
        elseif BC.v.low == "per"
            vLe_ext = [0; vLe]
            vRi_ext = [0; vRi]
        elseif BC.v.low == "pres"
            vLe_ext = [vLe(2); vLe] # zero gradient
            vRi_ext = [vRi(2); vRi] # zero gradient
        end
        if BC.v.up == "dir"
            vLe_ext = [vLe_ext; 2 * vLe[end] - vLe[end-1]]
            vRi_ext = [vRi_ext; 2 * vRi[1] - vRi(2)]
        elseif BC.v.up == "per"
            vLe_ext = [vLe_ext; 0]
            vRi_ext = [vRi_ext; 0]
        elseif BC.v.up == "pres"
            vLe_ext = [vLe_ext; vLe[end-1]] # zero gradient
            vRi_ext = [vRi_ext; vRi[end-1]] # zero gradient
        end
        ybc3 = kron(vLe_ext, Iv_uy_BC_lr3.ybc1) + kron(vRi_ext, Iv_uy_BC_lr3.ybc2)
        yIv_uy_lr3 = Iv_uy_BC_lr3.Bbc * ybc3

        ybc3 = kron(Iv_uy_BC_lu3.ybc1, vLo_i) + kron(Iv_uy_BC_lu3.ybc2, vUp_i)
        yIv_uy_lu3 = Iv_uy_BC_lr3.B2D * Iv_uy_BC_lu3.Bbc * ybc3
        yIv_uy3 = yIv_uy_lr3 + yIv_uy_lu3
    end

    # Iu_vx (low/up)
    ybc = kron(Iu_vx_BC_lu.ybc1, uLo) + kron(Iu_vx_BC_lu.ybc2, uUp)
    yIu_vx_lu = Iu_vx_BC_lu.Bbc * ybc
    # Iu_vx (left/right)
    # uLe_i = interp1(y, uLe, yp);
    # uRi_i = interp1(y, uRi, yp);
    ybc = kron(uLe_i, Iu_vx_BC_lr.ybc1) + kron(uRi_i, Iu_vx_BC_lr.ybc2)
    yIu_vx_lr = Iu_vx_BC_lu.B2D * Iu_vx_BC_lr.Bbc * ybc
    yIu_vx = yIu_vx_lr + yIu_vx_lu

    if order4
        if BC.u.left == "dir"
            uLo_ext = [2 * uLo[1] - uLo(2); uLo]
            uUp_ext = [2 * uUp[1] - uUp(2); uUp]
        elseif BC.u.left == "per"
            uLo_ext = [0; uLo]
            uUp_ext = [0; uUp]
        elseif BC.u.left == "pres"
            uLo_ext = [uLo(2); uLo] # zero gradient
            uUp_ext = [uUp(2); uUp] # zero gradient
        end
        if BC.u.right == "dir"
            uLo_ext = [uLo_ext; 2 * uLo[end] - uLo[end-1]]
            uUp_ext = [uUp_ext; 2 * uUp[1] - uUp(2)]
        elseif BC.u.right == "per"
            uLo_ext = [uLo_ext; 0]
            uUp_ext = [uUp_ext; 0]
        elseif BC.u.right == "pres"
            uLo_ext = [uLo_ext; uLo[end-1]] # zero gradient
            uUp_ext = [uUp_ext; uUp[end-1]] # zero gradient
        end
        ybc3 = kron(Iu_vx_BC_lu3.ybc1, uLo_ext) + kron(Iu_vx_BC_lu3.ybc2, uUp_ext)
        yIu_vx_lu3 = Iu_vx_BC_lu3.Bbc * ybc3

        ybc3 = kron(uLe_i, Iu_vx_BC_lr3.ybc1) + kron(uRi_i, Iu_vx_BC_lr3.ybc2)
        yIu_vx_lr3 = Iu_vx_BC_lu3.B2D * Iu_vx_BC_lr3.Bbc * ybc3
        yIu_vx3 = yIu_vx_lr3 + yIu_vx_lu3
    end

    # Iv_vy
    # vLo_i = interp1(x, vLo, xp);
    # vUp_i = interp1(x, vUp, xp);
    ybc = kron(Iv_vy_BC.ybc1, vLo_i) + kron(Iv_vy_BC.ybc2, vUp_i)
    yIv_vy = Iv_vy_BC.Bbc * ybc
    if order4
        ybc3 = kron(Iv_vy_BC3.ybc1, vLo_i) + kron(Iv_vy_BC3.ybc2, vUp_i)
        yIv_vy3 = Iv_vy_BC3.Bbc * ybc3
    end

    setup.discretization.yIu_ux = yIu_ux
    setup.discretization.yIv_uy = yIv_uy
    setup.discretization.yIu_vx = yIu_vx
    setup.discretization.yIv_vy = yIv_vy

    if order4
        setup.discretization.yIu_ux3 = yIu_ux3
        setup.discretization.yIv_uy3 = yIv_uy3
        setup.discretization.yIu_vx3 = yIu_vx3
        setup.discretization.yIv_vy3 = yIv_vy3
    end

    if visc ∈ ["qr", "LES", "ML"]

        # set BC for turbulent viscosity nu_t
        # in the periodic case, the value of nu_t is not needed
        # in all other cases, homogeneous (zero) Neumann conditions are used

        nuLe = zeros(Npy)
        nuRi = zeros(Npy)
        nuLo = zeros(Npx)
        nuUp = zeros(Npx)

        ## nu_ux
        Anu_ux_BC = setup.discretization.Anu_ux_BC
        ybc = kron(nuLe, Anu_ux_BC.ybc1) + kron(nuRi, Anu_ux_BC.ybc2)
        yAnu_ux = Anu_ux_BC.Bbc * ybc

        ## nu_uy
        Anu_uy_BC_lr = setup.discretization.Anu_uy_BC_lr
        Anu_uy_BC_lu = setup.discretization.Anu_uy_BC_lu

        nuLe_i = [nuLe[1]; nuLe; nuLe[end]]
        nuRi_i = [nuRi[1]; nuRi; nuRi[end]]
        # in x-direction
        ybc = kron(nuLe_i, Anu_uy_BC_lr.ybc1) + kron(nuRi_i, Anu_uy_BC_lr.ybc2)
        yAnu_uy_lr = Anu_uy_BC_lr.B2D * ybc

        # in y-direction
        ybc = kron(Anu_uy_BC_lu.ybc1, nuLo) + kron(Anu_uy_BC_lu.ybc2, nuUp)
        yAnu_uy_lu = Anu_uy_BC_lu.B2D * ybc

        yAnu_uy = yAnu_uy_lu + yAnu_uy_lr

        ## nu_vx
        Anu_vx_BC_lr = setup.discretization.Anu_vx_BC_lr
        Anu_vx_BC_lu = setup.discretization.Anu_vx_BC_lu

        nuLo_i = [nuLo[1]; nuLo; nuLo[end]]
        nuUp_i = [nuUp[1]; nuUp; nuUp[end]]

        # in y-direction
        ybc = kron(Anu_vx_BC_lu.ybc1, nuLo_i) + kron(Anu_vx_BC_lu.ybc2, nuUp_i)
        yAnu_vx_lu = Anu_vx_BC_lu.B2D * ybc
        # in x-direction
        ybc = kron(nuLe, Anu_vx_BC_lr.ybc1) + kron(nuRi, Anu_vx_BC_lr.ybc2)
        yAnu_vx_lr = Anu_vx_BC_lr.B2D * ybc

        yAnu_vx = yAnu_vx_lu + yAnu_vx_lr

        ## nu_vy
        Anu_vy_BC = setup.discretization.Anu_vy_BC
        ybc = kron(Anu_vy_BC.ybc1, nuLo) + kron(Anu_vy_BC.ybc2, nuUp)
        yAnu_vy = Anu_vy_BC.Bbc * ybc

        setup.discretization.yAnu_ux = yAnu_ux
        setup.discretization.yAnu_uy = yAnu_uy
        setup.discretization.yAnu_vx = yAnu_vx
        setup.discretization.yAnu_vy = yAnu_vy

        # set BC for getting du/dx, du/dy, dv/dx, dv/dy at cell centers

        uLo_p = uBC(xp, y[1], t, setup)
        uUp_p = uBC(xp, y[end], t, setup)

        vLe_p = vBC(x[1], yp, t, setup)
        vRi_p = vBC(x[end], yp, t, setup)

        Cux_k_BC = setup.discretization.Cux_k_BC
        ybc = kron(uLe_i, Cux_k_BC.ybc1) + kron(uRi_i, Cux_k_BC.ybc2)
        yCux_k = Cux_k_BC.Bbc * ybc

        Auy_k_BC = setup.discretization.Auy_k_BC
        ybc = kron(uLe_i, Auy_k_BC.ybc1) + kron(uRi_i, Auy_k_BC.ybc2)
        yAuy_k = Auy_k_BC.Bbc * ybc
        Cuy_k_BC = setup.discretization.Cuy_k_BC
        ybc = kron(Cuy_k_BC.ybc1, uLo_p) + kron(Cuy_k_BC.ybc2, uUp_p)
        yCuy_k = Cuy_k_BC.Bbc * ybc

        Avx_k_BC = setup.discretization.Avx_k_BC
        ybc = kron(Avx_k_BC.ybc1, vLo_i) + kron(Avx_k_BC.ybc2, vUp_i)
        yAvx_k = Avx_k_BC.Bbc * ybc

        Cvx_k_BC = setup.discretization.Cvx_k_BC
        ybc = kron(vLe_p, Cvx_k_BC.ybc1) + kron(vRi_p, Cvx_k_BC.ybc2)
        yCvx_k = Cvx_k_BC.Bbc * ybc

        Cvy_k_BC = setup.discretization.Cvy_k_BC
        ybc = kron(Cvy_k_BC.ybc1, vLo_i) + kron(Cvy_k_BC.ybc2, vUp_i)
        yCvy_k = Cvy_k_BC.Bbc * ybc

        setup.discretization.yCux_k = yCux_k
        setup.discretization.yCuy_k = yCuy_k
        setup.discretization.yCvx_k = yCvx_k
        setup.discretization.yCvy_k = yCvy_k
        setup.discretization.yAuy_k = yAuy_k
        setup.discretization.yAvx_k = yAvx_k
    end

    setup
end
