"""
    set_bc_vectors!(setup, t)

Construct boundary conditions
"""
function set_bc_vectors!(setup, t)
    @unpack is_steady, visc = setup.case
    @unpack Re = setup.fluid
    @unpack u_bc, v_bc, dudt_bc, dvdt_bc = setup.bc
    @unpack pLe, pRi, pLo, pUp = setup.bc
    @unpack Nux_in, Nvy_in, Np, Npx, Npy = setup.grid
    @unpack xin, yin, x, y, hx, hy, xp, yp = setup.grid
    @unpack order4 = setup.discretization
    @unpack Dux, Duy, Dvx, Dvy = setup.discretization
    @unpack Au_ux_bc, Au_uy_bc, Av_vx_bc, Av_vy_bc = setup.discretization
    @unpack Su_ux_bc, Su_uy_bc, Sv_vx_bc, Sv_vy_bc = setup.discretization
    @unpack Iu_ux_bc, Iv_uy_bc_lr, Iv_uy_bc_lu = setup.discretization
    @unpack Iu_vx_bc_lr, Iu_vx_bc_lu, Iv_vy_bc = setup.discretization
    @unpack Mx_bc, My_bc = setup.discretization
    @unpack Anu_vy_bc = setup.discretization
    @unpack Cux_k_bc, Cuy_k_bc, Cvx_k_bc, Cvy_k_bc, Auy_k_bc, Avx_k_bc =
        setup.discretization
    @unpack Su_vx_bc_lr, Su_vx_bc_lu, Sv_uy_bc_lr, Sv_uy_bc_lu = setup.discretization

    if order4
        @unpack α = setup.discretization
        @unpack Au_ux_bc3, Au_uy_bc3, Av_vx_bc3, Av_vy_bc3 = setup.discretization
        @unpack Iu_ux_bc3, Iv_uy_bc_lu3, Iv_uy_bc_lr3 = setup.discretization
        @unpack Iu_vx_bc_lu3, Iu_vx_bc_lr3, Iv_vy_bc3 = setup.discretization
        @unpack Su_ux_bc3, Su_uy_bc3, Sv_vx_bc3, Sv_vy_bc3 = setup.discretization
        @unpack Diffux_div, Diffuy_div, Diffvx_div, Diffvy_div = setup.discretization
        @unpack Mx_bc3, My_bc3 = setup.discretization
    end

    ## get BC values
    uLo = u_bc.(x, y[1], t, [setup])
    uUp = u_bc.(x, y[end], t, [setup])

    uLo_i = u_bc.(xin, y[1], t, [setup])
    uUp_i = u_bc.(xin, y[end], t, [setup])
    uLe_i = u_bc.(x[1], yp, t, [setup])
    uRi_i = u_bc.(x[end], yp, t, [setup])

    vLe = v_bc.(x[1], y, t, [setup])
    vRi = v_bc.(x[end], y, t, [setup])

    vLo_i = v_bc.(xp, y[1], t, [setup])
    vUp_i = v_bc.(xp, y[end], t, [setup])
    vLe_i = v_bc.(x[1], yin, t, [setup])
    vRi_i = v_bc.(x[end], yin, t, [setup])

    if !is_steady && setup.bc.bc_unsteady
        dudtLe_i = dudt_bc.(x[1], setup.grid.yp, t, [setup])
        dudtRi_i = dudt_bc.(x[end], setup.grid.yp, t, [setup])
        dvdtLo_i = dvdt_bc.(setup.grid.xp, y[1], t, [setup])
        dvdtUp_i = dvdt_bc.(setup.grid.xp, y[end], t, [setup])
    end

    ## boundary conditions for divergence

    # Mx
    ybc = kron(uLe_i, Mx_bc.ybc1) + kron(uRi_i, Mx_bc.ybc2)
    yMx = Mx_bc.Bbc * ybc
    if order4
        ybc3 = kron(uLe_i, Mx_bc3.ybc1) + kron(uRi_i, Mx_bc3.ybc2)
        yMx3 = Mx_bc3.Bbc * ybc3
        yMx = α * yMx - yMx3
    end

    # My
    ybc = kron(My_bc.ybc1, vLo_i) + kron(My_bc.ybc2, vUp_i)
    yMy = My_bc.Bbc * ybc
    if order4
        ybc3 = kron(My_bc3.ybc1, vLo_i) + kron(My_bc3.ybc2, vUp_i)
        yMy3 = My_bc3.Bbc * ybc3
        yMy = α * yMy - yMy3
    end

    yM = yMx + yMy
    setup.discretization.yM = yM

    # time derivative of divergence
    if !is_steady
        if setup.bc.bc_unsteady
            ybc = kron(dudtLe_i, Mx_bc.ybc1) + kron(dudtRi_i, Mx_bc.ybc2)
            ydMx = Mx_bc.Bbc * ybc
            if order4
                ybc3 = kron(dudtLe_i, Mx_bc3.ybc1) + kron(dudtRi_i, Mx_bc3.ybc2)
                ydMx3 = Mx_bc3.Bbc * ybc3
                ydMx = α * ydMx - ydMx3
            end

            # My
            ybc = kron(My_bc.ybc1, dvdtLo_i) + kron(My_bc.ybc2, dvdtUp_i)
            ydMy = My_bc.Bbc * ybc
            if order4
                ybc3 = kron(My_bc3.ybc1, dvdtLo_i) + kron(My_bc3.ybc2, dvdtUp_i)
                ydMy3 = My_bc3.Bbc * ybc3
                ydMy = α * ydMy - ydMy3
            end

            ydM = ydMx + ydMy
        else
            ydM = zeros(Np)
        end
        setup.discretization.ydM = ydM
    end

    ## boundary conditions for pressure

    # left and right side
    y1D_le = zeros(Nux_in)
    y1D_ri = zeros(Nux_in)
    if setup.bc.u.left == "pres"
        y1D_le[1] = -1
    end
    if setup.bc.u.right == "pres"
        y1D_ri[end] = 1
    end
    y_px = kron(hy .* pLe, y1D_le) + kron(hy .* pRi, y1D_ri)

    # lower and upper side
    y1D_lo = zeros(Nvy_in)
    y1D_up = zeros(Nvy_in)
    if setup.bc.v.low == "pres"
        y1D_lo[1] = -1
    end
    if setup.bc.v.up == "pres"
        y1D_up[end] = 1
    end
    y_py = kron(y1D_lo, hx .* pLo) + kron(y1D_up, hx .* pUp)

    setup.discretization.y_px = y_px
    setup.discretization.y_py = y_py

    ## boundary conditions for averaging

    # Au_ux
    ybc = kron(uLe_i, Au_ux_bc.ybc1) + kron(uRi_i, Au_ux_bc.ybc2)
    yAu_ux = Au_ux_bc.Bbc * ybc
    if order4
        ybc3 = kron(uLe_i, Au_ux_bc3.ybc1) + kron(uRi_i, Au_ux_bc3.ybc2)
        yAu_ux3 = Au_ux_bc3.Bbc * ybc3
    end

    # Au_uy
    ybc = kron(Au_uy_bc.ybc1, uLo_i) + kron(Au_uy_bc.ybc2, uUp_i)
    yAu_uy = Au_uy_bc.Bbc * ybc
    if order4
        ybc3 = kron(Au_uy_bc3.ybc1, uLo_i) + kron(Au_uy_bc3.ybc2, uUp_i)
        yAu_uy3 = Au_uy_bc3.Bbc * ybc3
    end

    # Av_vx
    ybc = kron(vLe_i, Av_vx_bc.ybc1) + kron(vRi_i, Av_vx_bc.ybc2)
    yAv_vx = Av_vx_bc.Bbc * ybc
    if order4
        ybc3 = kron(vLe_i, Av_vx_bc3.ybc1) + kron(vRi_i, Av_vx_bc3.ybc2)
        yAv_vx3 = Av_vx_bc3.Bbc * ybc3
    end

    # Av_vy
    ybc = kron(Av_vy_bc.ybc1, vLo_i) + kron(Av_vy_bc.ybc2, vUp_i)
    yAv_vy = Av_vy_bc.Bbc * ybc
    if order4
        ybc3 = kron(Av_vy_bc3.ybc1, vLo_i) + kron(Av_vy_bc3.ybc2, vUp_i)
        yAv_vy3 = Av_vy_bc3.Bbc * ybc3
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
    if order4
        ybc1 = kron(uLe_i, Su_ux_bc.ybc1) + kron(uRi_i, Su_ux_bc.ybc2)
        ybc3 = kron(uLe_i, Su_ux_bc3.ybc1) + kron(uRi_i, Su_ux_bc3.ybc2)
        ySu_ux = α * Su_ux_bc.Bbc * ybc1 - Su_ux_bc3.Bbc * ybc3

        ybc1 = kron(Su_uy_bc.ybc1, uLo_i) + kron(Su_uy_bc.ybc2, uUp_i)
        ybc3 = kron(Su_uy_bc3.ybc1, uLo_i) + kron(Su_uy_bc3.ybc2, uUp_i)
        ySu_uy = α * Su_uy_bc.Bbc * ybc1 - Su_uy_bc3.Bbc * ybc3

        ybc1 = kron(vLe_i, Sv_vx_bc.ybc1) + kron(vRi_i, Sv_vx_bc.ybc2)
        ybc3 = kron(vLe_i, Sv_vx_bc3.ybc1) + kron(vRi_i, Sv_vx_bc3.ybc2)
        ySv_vx = α * Sv_vx_bc.Bbc * ybc1 - Sv_vx_bc3.Bbc * ybc3

        ybc1 = kron(Sv_vy_bc.ybc1, vLo_i) + kron(Sv_vy_bc.ybc2, vUp_i)
        ybc3 = kron(Sv_vy_bc3.ybc1, vLo_i) + kron(Sv_vy_bc3.ybc2, vUp_i)
        ySv_vy = α * Sv_vy_bc.Bbc * ybc1 - Sv_vy_bc3.Bbc * ybc3

        if visc == "laminar"
            yDiffu = 1 / Re * (Diffux_div * ySu_ux + Diffuy_div * ySu_uy)
            yDiffv = 1 / Re * (Diffvx_div * ySv_vx + Diffvy_div * ySv_vy)
            setup.discretization.yDiffu = yDiffu
            setup.discretization.yDiffv = yDiffv
        elseif visc ∈ ["keps", "LES", "qr", "ML"]
            error("fourth order turbulent diffusion not implemented")
        end
    else
        # Su_ux
        ybc = kron(uLe_i, Su_ux_bc.ybc1) + kron(uRi_i, Su_ux_bc.ybc2)
        ySu_ux = Su_ux_bc.Bbc * ybc

        # Su_uy
        ybc = kron(Su_uy_bc.ybc1, uLo_i) + kron(Su_uy_bc.ybc2, uUp_i)
        ySu_uy = Su_uy_bc.Bbc * ybc

        Sv_uy = Sv_uy_bc_lr.B2D * Sv_uy_bc_lu.B2D

        # Sv_uy (left/right)
        ybc = kron(vLe, Sv_uy_bc_lr.ybc1) + kron(vRi, Sv_uy_bc_lr.ybc2)
        ySv_uy_lr = Sv_uy_bc_lr.Bbc * ybc
        # Iv_uy (low/up)
        ybc = kron(Sv_uy_bc_lu.ybc1, vLo_i) + kron(Sv_uy_bc_lu.ybc2, vUp_i)
        ySv_uy_lu = Sv_uy_bc_lr.B2D * Sv_uy_bc_lu.Bbc * ybc

        ySv_uy = ySv_uy_lr + ySv_uy_lu

        # Su_vx (low/up)
        ybc = kron(Su_vx_bc_lu.ybc1, uLo) + kron(Su_vx_bc_lu.ybc2, uUp)
        ySu_vx_lu = Su_vx_bc_lu.Bbc * ybc
        # Su_vx (left/right)
        ybc = kron(uLe_i, Su_vx_bc_lr.ybc1) + kron(uRi_i, Su_vx_bc_lr.ybc2)
        ySu_vx_lr = Su_vx_bc_lu.B2D * Su_vx_bc_lr.Bbc * ybc
        ySu_vx = ySu_vx_lr + ySu_vx_lu

        # Sv_vx
        ybc = kron(vLe_i, Sv_vx_bc.ybc1) + kron(vRi_i, Sv_vx_bc.ybc2)
        ySv_vx = Sv_vx_bc.Bbc * ybc

        # Sv_vy
        ybc = kron(Sv_vy_bc.ybc1, vLo_i) + kron(Sv_vy_bc.ybc2, vUp_i)
        ySv_vy = Sv_vy_bc.Bbc * ybc

        if visc == "laminar"
            yDiffu = Dux * (1 / Re * ySu_ux) + Duy * (1 / Re * ySu_uy)
            yDiffv = Dvx * (1 / Re * ySv_vx) + Dvy * (1 / Re * ySv_vy)
            setup.discretization.yDiffu = yDiffu
            setup.discretization.yDiffv = yDiffv
        elseif visc ∈ ["keps", "LES", "qr", "ML"]
            # instead, we will use the following values directly (see diffusion.jl and strain_tensor.jl)
            setup.discretization.ySu_ux = ySu_ux
            setup.discretization.ySu_uy = ySu_uy
            setup.discretization.ySu_vx = ySu_vx
            setup.discretization.ySv_vx = ySv_vx
            setup.discretization.ySv_vy = ySv_vy
            setup.discretization.ySv_uy = ySv_uy
        end
    end

    ## boundary conditions for interpolation

    # Iu_ux
    ybc = kron(uLe_i, Iu_ux_bc.ybc1) + kron(uRi_i, Iu_ux_bc.ybc2)
    yIu_ux = Iu_ux_bc.Bbc * ybc
    if order4
        ybc3 = kron(uLe_i, Iu_ux_bc3.ybc1) + kron(uRi_i, Iu_ux_bc3.ybc2)
        yIu_ux3 = Iu_ux_bc3.Bbc * ybc3
    end

    # Iv_uy (left/right)
    ybc = kron(vLe, Iv_uy_bc_lr.ybc1) + kron(vRi, Iv_uy_bc_lr.ybc2)
    yIv_uy_lr = Iv_uy_bc_lr.Bbc * ybc
    # Iv_uy (low/up)
    ybc = kron(Iv_uy_bc_lu.ybc1, vLo_i) + kron(Iv_uy_bc_lu.ybc2, vUp_i)
    yIv_uy_lu = Iv_uy_bc_lr.B2D * Iv_uy_bc_lu.Bbc * ybc
    yIv_uy = yIv_uy_lr + yIv_uy_lu

    if order4
        if setup.bc.v.low == "dir"
            vLe_ext = [2 * vLe[1] - vLe[2]; vLe]
            vRi_ext = [2 * vRi[1] - vRi[2]; vRi]
        elseif setup.bc.v.low == "per"
            vLe_ext = [0; vLe]
            vRi_ext = [0; vRi]
        elseif setup.bc.v.low == "pres"
            vLe_ext = [vLe[2]; vLe] # zero gradient
            vRi_ext = [vRi[2]; vRi] # zero gradient
        end
        if setup.bc.v.up == "dir"
            vLe_ext = [vLe_ext; 2 * vLe[end] - vLe[end-1]]
            vRi_ext = [vRi_ext; 2 * vRi[1] - vRi[2]]
        elseif setup.bc.v.up == "per"
            vLe_ext = [vLe_ext; 0]
            vRi_ext = [vRi_ext; 0]
        elseif setup.bc.v.up == "pres"
            vLe_ext = [vLe_ext; vLe[end-1]] # zero gradient
            vRi_ext = [vRi_ext; vRi[end-1]] # zero gradient
        end
        ybc3 = kron(vLe_ext, Iv_uy_bc_lr3.ybc1) + kron(vRi_ext, Iv_uy_bc_lr3.ybc2)
        yIv_uy_lr3 = Iv_uy_bc_lr3.Bbc * ybc3

        ybc3 = kron(Iv_uy_bc_lu3.ybc1, vLo_i) + kron(Iv_uy_bc_lu3.ybc2, vUp_i)
        yIv_uy_lu3 = Iv_uy_bc_lr3.B2D * Iv_uy_bc_lu3.Bbc * ybc3
        yIv_uy3 = yIv_uy_lr3 + yIv_uy_lu3
    end

    # Iu_vx (low/up)
    ybc = kron(Iu_vx_bc_lu.ybc1, uLo) + kron(Iu_vx_bc_lu.ybc2, uUp)
    yIu_vx_lu = Iu_vx_bc_lu.Bbc * ybc
    # Iu_vx (left/right)
    ybc = kron(uLe_i, Iu_vx_bc_lr.ybc1) + kron(uRi_i, Iu_vx_bc_lr.ybc2)
    yIu_vx_lr = Iu_vx_bc_lu.B2D * Iu_vx_bc_lr.Bbc * ybc
    yIu_vx = yIu_vx_lr + yIu_vx_lu

    if order4
        if setup.bc.u.left == "dir"
            uLo_ext = [2 * uLo[1] - uLo[2]; uLo]
            uUp_ext = [2 * uUp[1] - uUp[2]; uUp]
        elseif setup.bc.u.left == "per"
            uLo_ext = [0; uLo]
            uUp_ext = [0; uUp]
        elseif setup.bc.u.left == "pres"
            uLo_ext = [uLo[2]; uLo] # zero gradient
            uUp_ext = [uUp[2]; uUp] # zero gradient
        end
        if setup.bc.u.right == "dir"
            uLo_ext = [uLo_ext; 2 * uLo[end] - uLo[end-1]]
            uUp_ext = [uUp_ext; 2 * uUp[1] - uUp[2]]
        elseif setup.bc.u.right == "per"
            uLo_ext = [uLo_ext; 0]
            uUp_ext = [uUp_ext; 0]
        elseif setup.bc.u.right == "pres"
            uLo_ext = [uLo_ext; uLo[end-1]] # zero gradient
            uUp_ext = [uUp_ext; uUp[end-1]] # zero gradient
        end
        ybc3 = kron(Iu_vx_bc_lu3.ybc1, uLo_ext) + kron(Iu_vx_bc_lu3.ybc2, uUp_ext)
        yIu_vx_lu3 = Iu_vx_bc_lu3.Bbc * ybc3

        ybc3 = kron(uLe_i, Iu_vx_bc_lr3.ybc1) + kron(uRi_i, Iu_vx_bc_lr3.ybc2)
        yIu_vx_lr3 = Iu_vx_bc_lu3.B2D * Iu_vx_bc_lr3.Bbc * ybc3
        yIu_vx3 = yIu_vx_lr3 + yIu_vx_lu3
    end

    # Iv_vy
    # vLo_i = interp1(x, vLo, xp);
    # vUp_i = interp1(x, vUp, xp);
    ybc = kron(Iv_vy_bc.ybc1, vLo_i) + kron(Iv_vy_bc.ybc2, vUp_i)
    yIv_vy = Iv_vy_bc.Bbc * ybc
    if order4
        ybc3 = kron(Iv_vy_bc3.ybc1, vLo_i) + kron(Iv_vy_bc3.ybc2, vUp_i)
        yIv_vy3 = Iv_vy_bc3.Bbc * ybc3
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
        Anu_ux_bc = setup.discretization.Anu_ux_bc
        ybc = kron(nuLe, Anu_ux_bc.ybc1) + kron(nuRi, Anu_ux_bc.ybc2)
        yAnu_ux = Anu_ux_bc.Bbc * ybc

        ## nu_uy
        Anu_uy_bc_lr = setup.discretization.Anu_uy_bc_lr
        Anu_uy_bc_lu = setup.discretization.Anu_uy_bc_lu

        nuLe_i = [nuLe[1]; nuLe; nuLe[end]]
        nuRi_i = [nuRi[1]; nuRi; nuRi[end]]
        # in x-direction
        ybc = kron(nuLe_i, Anu_uy_bc_lr.ybc1) + kron(nuRi_i, Anu_uy_bc_lr.ybc2)
        yAnu_uy_lr = Anu_uy_bc_lr.B2D * ybc

        # in y-direction
        ybc = kron(Anu_uy_bc_lu.ybc1, nuLo) + kron(Anu_uy_bc_lu.ybc2, nuUp)
        yAnu_uy_lu = Anu_uy_bc_lu.B2D * ybc

        yAnu_uy = yAnu_uy_lu + yAnu_uy_lr

        ## nu_vx
        Anu_vx_bc_lr = setup.discretization.Anu_vx_bc_lr
        Anu_vx_bc_lu = setup.discretization.Anu_vx_bc_lu

        nuLo_i = [nuLo[1]; nuLo; nuLo[end]]
        nuUp_i = [nuUp[1]; nuUp; nuUp[end]]

        # in y-direction
        ybc = kron(Anu_vx_bc_lu.ybc1, nuLo_i) + kron(Anu_vx_bc_lu.ybc2, nuUp_i)
        yAnu_vx_lu = Anu_vx_bc_lu.B2D * ybc
        # in x-direction
        ybc = kron(nuLe, Anu_vx_bc_lr.ybc1) + kron(nuRi, Anu_vx_bc_lr.ybc2)
        yAnu_vx_lr = Anu_vx_bc_lr.B2D * ybc

        yAnu_vx = yAnu_vx_lu + yAnu_vx_lr

        ## nu_vy
        ybc = kron(Anu_vy_bc.ybc1, nuLo) + kron(Anu_vy_bc.ybc2, nuUp)
        yAnu_vy = Anu_vy_bc.Bbc * ybc

        setup.discretization.yAnu_ux = yAnu_ux
        setup.discretization.yAnu_uy = yAnu_uy
        setup.discretization.yAnu_vx = yAnu_vx
        setup.discretization.yAnu_vy = yAnu_vy

        # set BC for getting du/dx, du/dy, dv/dx, dv/dy at cell centers

        uLo_p = u_bc.(xp, y[1], t, [setup])
        uUp_p = u_bc.(xp, y[end], t, [setup])

        vLe_p = v_bc.(x[1], yp, t, [setup])
        vRi_p = v_bc.(x[end], yp, t, [setup])

        ybc = kron(uLe_i, Cux_k_bc.ybc1) + kron(uRi_i, Cux_k_bc.ybc2)
        yCux_k = Cux_k_bc.Bbc * ybc

        ybc = kron(uLe_i, Auy_k_bc.ybc1) + kron(uRi_i, Auy_k_bc.ybc2)
        yAuy_k = Auy_k_bc.Bbc * ybc

        ybc = kron(Cuy_k_bc.ybc1, uLo_p) + kron(Cuy_k_bc.ybc2, uUp_p)
        yCuy_k = Cuy_k_bc.Bbc * ybc

        ybc = kron(Avx_k_bc.ybc1, vLo_i) + kron(Avx_k_bc.ybc2, vUp_i)
        yAvx_k = Avx_k_bc.Bbc * ybc

        ybc = kron(vLe_p, Cvx_k_bc.ybc1) + kron(vRi_p, Cvx_k_bc.ybc2)
        yCvx_k = Cvx_k_bc.Bbc * ybc

        ybc = kron(Cvy_k_bc.ybc1, vLo_i) + kron(Cvy_k_bc.ybc2, vUp_i)
        yCvy_k = Cvy_k_bc.Bbc * ybc

        setup.discretization.yCux_k = yCux_k
        setup.discretization.yCuy_k = yCuy_k
        setup.discretization.yCvx_k = yCvx_k
        setup.discretization.yCvy_k = yCvy_k
        setup.discretization.yAuy_k = yAuy_k
        setup.discretization.yAvx_k = yAvx_k
    end

    setup
end
