"""
    set_bc_vectors!(setup, t)

Construct boundary conditions.
"""
function set_bc_vectors!(setup, t)
    @unpack problem = setup.case
    @unpack model = setup
    @unpack Re = setup.fluid
    @unpack u_bc, v_bc, dudt_bc, dvdt_bc = setup.bc
    @unpack pLe, pRi, pLo, pUp, bc_unsteady = setup.bc
    @unpack Nux_in, Nvy_in, Np, Npx, Npy = setup.grid
    @unpack xin, yin, x, y, hx, hy, xp, yp = setup.grid
    @unpack order4 = setup.discretization
    @unpack Dux, Duy, Dvx, Dvy = setup.discretization
    @unpack Au_ux_bc, Au_uy_bc, Av_vx_bc, Av_vy_bc = setup.discretization
    @unpack Su_ux_bc, Su_uy_bc, Sv_vx_bc, Sv_vy_bc = setup.discretization
    @unpack Iu_ux_bc, Iv_uy_bc_lr, Iv_uy_bc_lu = setup.discretization
    @unpack Iu_vx_bc_lr, Iu_vx_bc_lu, Iv_vy_bc = setup.discretization
    @unpack Mx_bc, My_bc = setup.discretization
    @unpack Aν_vy_bc = setup.discretization
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

    ## Get BC values
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

    if !is_steady(problem) && bc_unsteady
        dudtLe_i = dudt_bc.(x[1], yp, t, [setup])
        dudtRi_i = dudt_bc.(x[end], yp, t, [setup])
        dvdtLo_i = dvdt_bc.(xp, y[1], t, [setup])
        dvdtUp_i = dvdt_bc.(xp, y[end], t, [setup])
    end

    ## Boundary conditions for divergence

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
    @pack! setup.discretization = yM

    # Time derivative of divergence
    if !is_steady(problem)
        if bc_unsteady
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
        @pack! setup.discretization = ydM
    end

    ## Boundary conditions for pressure

    # Left and right side
    y1D_le = zeros(Nux_in)
    y1D_ri = zeros(Nux_in)
    if setup.bc.u.left == :pressure
        y1D_le[1] = -1
    end
    if setup.bc.u.right == :pressure
        y1D_ri[end] = 1
    end
    y_px = kron(hy .* pLe, y1D_le) + kron(hy .* pRi, y1D_ri)

    # Lower and upper side
    y1D_lo = zeros(Nvy_in)
    y1D_up = zeros(Nvy_in)
    if setup.bc.v.low == :pressure
        y1D_lo[1] = -1
    end
    if setup.bc.v.up == :pressure
        y1D_up[end] = 1
    end
    y_py = kron(y1D_lo, hx .* pLo) + kron(y1D_up, hx .* pUp)

    y_p = [y_px; y_py]
    @pack! setup.discretization = y_p

    ## Boundary conditions for averaging
    # Au_ux
    ybc = kron(uLe_i, Au_ux_bc.ybc1) + kron(uRi_i, Au_ux_bc.ybc2)
    yAu_ux = Au_ux_bc.Bbc * ybc

    # Au_uy
    ybc = kron(Au_uy_bc.ybc1, uLo_i) + kron(Au_uy_bc.ybc2, uUp_i)
    yAu_uy = Au_uy_bc.Bbc * ybc

    # Av_vx
    ybc = kron(vLe_i, Av_vx_bc.ybc1) + kron(vRi_i, Av_vx_bc.ybc2)
    yAv_vx = Av_vx_bc.Bbc * ybc

    # Av_vy
    ybc = kron(Av_vy_bc.ybc1, vLo_i) + kron(Av_vy_bc.ybc2, vUp_i)
    yAv_vy = Av_vy_bc.Bbc * ybc

    @pack! setup.discretization = yAu_ux, yAu_uy, yAv_vx, yAv_vy

    if order4
        # Au_ux
        ybc3 = kron(uLe_i, Au_ux_bc3.ybc1) + kron(uRi_i, Au_ux_bc3.ybc2)
        yAu_ux3 = Au_ux_bc3.Bbc * ybc3

        # Au_uy
        ybc3 = kron(Au_uy_bc3.ybc1, uLo_i) + kron(Au_uy_bc3.ybc2, uUp_i)
        yAu_uy3 = Au_uy_bc3.Bbc * ybc3

        # Av_vx
        ybc3 = kron(vLe_i, Av_vx_bc3.ybc1) + kron(vRi_i, Av_vx_bc3.ybc2)
        yAv_vx3 = Av_vx_bc3.Bbc * ybc3

        # Av_vy
        ybc3 = kron(Av_vy_bc3.ybc1, vLo_i) + kron(Av_vy_bc3.ybc2, vUp_i)
        yAv_vy3 = Av_vy_bc3.Bbc * ybc3

        @pack! setup.discretization = yAu_ux3, yAu_uy3, yAv_vx3, yAv_vy3
    end

    ## Boundary conditions for diffusion
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

        if model isa LaminarModel
            yDiffu = 1 / Re * (Diffux_div * ySu_ux + Diffuy_div * ySu_uy)
            yDiffv = 1 / Re * (Diffvx_div * ySv_vx + Diffvy_div * ySv_vy)
            yDiff = [yDiffu; yDiffv]
            @pack! setup.discretization = yDiff
        else
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

        if model isa LaminarModel
            yDiffu = 1 / Re * (Dux * ySu_ux + Duy * ySu_uy)
            yDiffv = 1 / Re * (Dvx * ySv_vx + Dvy * ySv_vy)
            yDiff = [yDiffu; yDiffv]
            @pack! setup.discretization = yDiff
        else
            # Instead, we will use the following values directly (see diffusion.jl and strain_tensor.jl)
            @pack! setup.discretization = ySu_ux, ySu_uy, ySu_vx, ySv_vx, ySv_vy, ySv_uy
        end
    end

    ## Boundary conditions for interpolation

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
        if setup.bc.v.low == :dirichlet
            vLe_ext = [2 * vLe[1] - vLe[2]; vLe]
            vRi_ext = [2 * vRi[1] - vRi[2]; vRi]
        elseif setup.bc.v.low == :periodic
            vLe_ext = [0; vLe]
            vRi_ext = [0; vRi]
        elseif setup.bc.v.low == :pressure
            vLe_ext = [vLe[2]; vLe]
            vRi_ext = [vRi[2]; vRi]
        end
        if setup.bc.v.up == :dirichlet
            vLe_ext = [vLe_ext; 2 * vLe[end] - vLe[end-1]]
            vRi_ext = [vRi_ext; 2 * vRi[1] - vRi[2]]
        elseif setup.bc.v.up == :periodic
            vLe_ext = [vLe_ext; 0]
            vRi_ext = [vRi_ext; 0]
        elseif setup.bc.v.up == :pressure
            vLe_ext = [vLe_ext; vLe[end-1]]
            vRi_ext = [vRi_ext; vRi[end-1]]
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
        if setup.bc.u.left == :dirichlet
            uLo_ext = [2 * uLo[1] - uLo[2]; uLo]
            uUp_ext = [2 * uUp[1] - uUp[2]; uUp]
        elseif setup.bc.u.left == :periodic
            uLo_ext = [0; uLo]
            uUp_ext = [0; uUp]
        elseif setup.bc.u.left == :pressure
            uLo_ext = [uLo[2]; uLo]
            uUp_ext = [uUp[2]; uUp]
        end
        if setup.bc.u.right == :dirichlet
            uLo_ext = [uLo_ext; 2 * uLo[end] - uLo[end-1]]
            uUp_ext = [uUp_ext; 2 * uUp[1] - uUp[2]]
        elseif setup.bc.u.right == :periodic
            uLo_ext = [uLo_ext; 0]
            uUp_ext = [uUp_ext; 0]
        elseif setup.bc.u.right == :pressure
            uLo_ext = [uLo_ext; uLo[end-1]]
            uUp_ext = [uUp_ext; uUp[end-1]]
        end
        ybc3 = kron(Iu_vx_bc_lu3.ybc1, uLo_ext) + kron(Iu_vx_bc_lu3.ybc2, uUp_ext)
        yIu_vx_lu3 = Iu_vx_bc_lu3.Bbc * ybc3

        ybc3 = kron(uLe_i, Iu_vx_bc_lr3.ybc1) + kron(uRi_i, Iu_vx_bc_lr3.ybc2)
        yIu_vx_lr3 = Iu_vx_bc_lu3.B2D * Iu_vx_bc_lr3.Bbc * ybc3
        yIu_vx3 = yIu_vx_lr3 + yIu_vx_lu3
    end

    # Iv_vy
    ybc = kron(Iv_vy_bc.ybc1, vLo_i) + kron(Iv_vy_bc.ybc2, vUp_i)
    yIv_vy = Iv_vy_bc.Bbc * ybc
    if order4
        ybc3 = kron(Iv_vy_bc3.ybc1, vLo_i) + kron(Iv_vy_bc3.ybc2, vUp_i)
        yIv_vy3 = Iv_vy_bc3.Bbc * ybc3
    end

    @pack! setup.discretization = yIu_ux, yIv_uy, yIu_vx, yIv_vy

    if order4
        @pack! setup.discretization = yIu_ux3, yIv_uy3, yIu_vx3, yIv_vy3
    end

    if model isa Union{QRModel, SmagorinskyModel, MixingLengthModel}
        # Set BC for turbulent viscosity nu_t
        # In the periodic case, the value of nu_t is not needed
        # In all other cases, homogeneous (zero) Neumann conditions are used

        nuLe = zeros(Npy)
        nuRi = zeros(Npy)
        nuLo = zeros(Npx)
        nuUp = zeros(Npx)

        ## Nu_ux
        @unpack Aν_ux_bc = setup.discretization
        ybc = kron(nuLe, Aν_ux_bc.ybc1) + kron(nuRi, Aν_ux_bc.ybc2)
        yAν_ux = Aν_ux_bc.Bbc * ybc

        ## Nu_uy
        @unpack Aν_uy_bc_lr, Aν_uy_bc_lu = setup.discretization

        nuLe_i = [nuLe[1]; nuLe; nuLe[end]]
        nuRi_i = [nuRi[1]; nuRi; nuRi[end]]
        # In x-direction
        ybc = kron(nuLe_i, Aν_uy_bc_lr.ybc1) + kron(nuRi_i, Aν_uy_bc_lr.ybc2)
        yAν_uy_lr = Aν_uy_bc_lr.B2D * ybc

        # In y-direction
        ybc = kron(Aν_uy_bc_lu.ybc1, nuLo) + kron(Aν_uy_bc_lu.ybc2, nuUp)
        yAν_uy_lu = Aν_uy_bc_lu.B2D * ybc

        yAν_uy = yAν_uy_lu + yAν_uy_lr

        ## Nu_vx
        @unpack Aν_vx_bc_lr, Aν_vx_bc_lu = setup.discretization

        nuLo_i = [nuLo[1]; nuLo; nuLo[end]]
        nuUp_i = [nuUp[1]; nuUp; nuUp[end]]

        # In y-direction
        ybc = kron(Aν_vx_bc_lu.ybc1, nuLo_i) + kron(Aν_vx_bc_lu.ybc2, nuUp_i)
        yAν_vx_lu = Aν_vx_bc_lu.B2D * ybc
        # In x-direction
        ybc = kron(nuLe, Aν_vx_bc_lr.ybc1) + kron(nuRi, Aν_vx_bc_lr.ybc2)
        yAν_vx_lr = Aν_vx_bc_lr.B2D * ybc

        yAν_vx = yAν_vx_lu + yAν_vx_lr

        ## Nu_vy
        ybc = kron(Aν_vy_bc.ybc1, nuLo) + kron(Aν_vy_bc.ybc2, nuUp)
        yAν_vy = Aν_vy_bc.Bbc * ybc

        @pack! setup.discretization = yAν_ux, yAν_uy, yAν_vx, yAν_vy

        # Set BC for getting du/dx, du/dy, dv/dx, dv/dy at cell centers

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

        @pack! setup.discretization = yCux_k
        yCuy_k, yCvx_k, yCvy_k, yAuy_k, yAvx_k
    end

    setup
end
