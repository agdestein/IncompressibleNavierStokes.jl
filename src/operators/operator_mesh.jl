function operator_mesh!(setup)
    bc = setup.bc
    order4 = setup.discretization.order4
    α = setup.discretization.α

    ## Pressure volumes
    @unpack Nx, Ny, x, y, hx, hy, gx, gy, xp, yp = setup.grid
    # @unpack gx, gy = setup.grid

    # Number of pressure points
    Npx = Nx
    Npy = Ny
    Np = Npx * Npy

    ## U-volumes
    # X[1]   x[2]   x[3] ....      x[Nx]   x[Nx+1]
    # |      |      |              |       |
    # |      |      |              |       |
    # Dirichlet BC:
    # ULe    u[1]   u[2] ....      u(Nx-1) uRi
    # Periodic BC:
    # U[1]   u[2]   u[3] ....      u[Nx]   u[1]
    # Pressure BC:
    # U[1]   u[2]   u[3] ....      u[Nx]   u[Nx+1]

    # X-dir
    Nux_b = 2               # Boundary points
    Nux_in = Nx + 1            # Inner points
    if bc.u.left ∈ [:dirichlet, :symmetric]
        Nux_in -= 1
    end
    if bc.u.right ∈ [:dirichlet, :symmetric]
        Nux_in -= 1
    end
    if bc.u.left == :periodic && bc.u.right == :periodic
        Nux_in -= 1
    end
    Nux_t = Nux_in + Nux_b  # Total number

    # Y-dir
    Nuy_b = 2               # Boundary points
    Nuy_in = Ny              # Inner points
    Nuy_t = Nuy_in + Nuy_b  # Total number

    # Total number
    Nu = Nux_in * Nuy_in


    ## V-volumes

    # X-dir
    Nvx_b = 2               # Boundary points
    Nvx_in = Nx              # Inner points
    Nvx_t = Nvx_in + Nvx_b  # Total number

    # Y-dir
    Nvy_b = 2               # Boundary points
    Nvy_in = Ny + 1            # Inner points
    if bc.v.low == :dirichlet || bc.v.low == :symmetric
        Nvy_in = Nvy_in - 1
    end
    if bc.v.up == :dirichlet || bc.v.up == :symmetric
        Nvy_in = Nvy_in - 1
    end
    if bc.v.low == :periodic && bc.v.up == :periodic
        Nvy_in = Nvy_in - 1
    end
    Nvy_t = Nvy_in + Nvy_b  # Total number

    # Total number
    Nv = Nvx_in * Nvy_in

    # Total number of velocity points
    NV = Nu + Nv

    # Total number of unknowns
    Ntot = NV + Np

    ## Extra variables
    N1 = (Nux_in + 1) * Nuy_in # size(Iu_ux, 1);
    N2 = Nux_in * (Nuy_in + 1) # size(Iv_uy, 1);
    N3 = (Nvx_in + 1) * Nvy_in # Size(Iu_vx, 1);
    N4 = Nvx_in * (Nvy_in + 1) # Size(Iv_vy, 1);


    ## For a grid with three times larger volumes:
    if order4
        hx3 = zeros(Nx, 1)
        hx3[2:end-1] = hx[1:end-2] + hx[2:end-1] + hx[3:end]
        if bc.u.left == :periodic && bc.u.right == :periodic
            hx3[1] = hx[end] + hx[1] + hx[2]
            hx3[end] = hx[end-1] + hx[end] + hx[1]
        else
            hx3[1] = 2 * hx[1] + hx[2]
            hx3[end] = hx[end-1] + 2 * hx[end]
        end

        hy3 = zeros(Ny, 1)
        hy3[2:end-1] = hy[1:end-2] + hy[2:end-1] + hy[3:end]
        if bc.v.low == :periodic && bc.v.up == :periodic
            hy3[1] = hy[end] + hy[1] + hy[2]
            hy3[end] = hy[end-1] + hy[end] + hy[1]
        else
            hy3[1] = 2 * hy[1] + hy[2]
            hy3[end] = hy[end-1] + 2 * hy[end]
        end

        hxi3 = hx3
        hyi3 = hy3


        # Distance between pressure points
        gx3 = zeros(Nx + 1, 1)
        gx3[3:Nx-1] = gx[2:end-3] + gx[3:end-2] + gx[4:end-1]
        if bc.u.left == :periodic && bc.u.right == :periodic
            gx3[1] = gx[end-1] + gx[end] + gx[1] + gx[2]
            gx3[2] = gx[end] + gx[1] + gx[2] + gx[3]
            gx3[end-1] = gx[end-2] + gx[end-1] + gx[end] + gx[1]
            gx3[end] = gx[end-1] + gx[end] + gx[1] + gx[2]
        else
            gx3[1] = 2 * gx[1] + 2 * gx[2]
            gx3[2] = 2 * gx[1] + gx[2] + gx[3]
            gx3[end-1] = 2 * gx[end] + gx[end-1] + gx[end-2]
            gx3[end] = 2 * gx[end] + 2 * gx[end-1]
        end

        # Distance between pressure points
        gy3 = zeros(Ny + 1, 1)
        gy3[3:Ny-1] = gy[2:end-3] + gy[3:end-2] + gy[4:end-1]
        if bc.v.low == :periodic && bc.v.up == :periodic
            gy3[1] = gy[end-1] + gy[end] + gy[1] + gy[2]
            gy3[2] = gy[end] + gy[1] + gy[2] + gy[3]
            gy3[end-1] = gy[end-2] + gy[end-1] + gy[end] + gy[1]
            gy3[end] = gy[end-1] + gy[end] + gy[1] + gy[2]
        else
            gy3[1] = 2 * gy[1] + 2 * gy[2]
            gy3[2] = 2 * gy[1] + gy[2] + gy[3]
            gy3[end-1] = 2 * gy[end] + gy[end-1] + gy[end-2]
            gy3[end] = 2 * gy[end] + 2 * gy[end-1]
        end
    end

    ## Adapt mesh metrics depending on number of volumes

    ## X-direction

    # Gxd: differentiation
    gxd = gx
    gxd[1] = hx[1]
    gxd[end] = hx[end]

    # Hxi: integration and hxd: differentiation
    # Map to find suitable size
    hxi = hx

    # Restrict Nx+2 to Nux_in+1 points
    if bc.u.left == :dirichlet && bc.u.right == :dirichlet
        xin = x[2:end-1]
        hxd = hx
        gxi = gx[2:end-1]
        diagpos = 1

        if setup.discretization.order4
            hxd3 = [hx3[1]; hx3; hx3[end]]
            hxd13 = [hx[1]; hx; hx[end]]
            gxd3 = [2 * gx[1] + gx[2] + gx[3]; gx3; 2 * gx[end] + gx[end-1] + gx[end-2]]
            gxd13 = [gx[2]; 2 * gx[1]; gx[2:end-1]; 2 * gx[end]; gx[end-1]]
            gxi3 = gx3[2:end-1]
        end
    end

    if bc.u.left == :dirichlet && bc.u.right == :pressure
        xin = x[2:end]
        hxd = [hx; hx[end]]
        gxi = gx[2:end]
        diagpos = 1
    end

    if bc.u.left == :pressure && bc.u.right == :dirichlet
        xin = x[1:end-1]
        hxd = [hx[1]; hx]
        gxi = gx[1:end-1]
        diagpos = 0
    end

    if bc.u.left == :pressure && bc.u.right == :pressure
        xin = x[1:end]
        hxd = [hx[1]; hx; hx[end]]
        gxi = gx
        diagpos = 0
    end

    if bc.u.left == :periodic && bc.u.right == :periodic
        xin = x[1:end-1]
        hxd = [hx[end]; hx]
        gxi = [gx[1] + gx[end]; gx[2:end-1]]
        gxd[1] = (hx[1] + hx[end]) / 2
        gxd[end] = (hx[1] + hx[end]) / 2
        diagpos = 0

        if order4
            hxd3 = [hx3[end-1]; hx3[end]; hx3; hx3[1]]
            hxd13 = [hx[end-1]; hx[end]; hx; hx[1]]
            gxd3 = [gx3[end-1]; gx3; gx3[2]]
            gxd13 = [gx[end-1]; gx[1] + gx[end]; gx[2:end-1]; gx[end] + gx[1]; gx[2]]
            gxi3 = gx3[1:end-1]
        end
    end

    Bmap = spdiagm(Nux_in + 1, Nx + 2, diagpos => ones(Nux_in + 1))

    # Matrix to map from Nvx_t-1 to Nux_in points
    # (used in interpolation, convection_diffusion, viscosity)
    Bvux = spdiagm(Nux_in, Nvx_t - 1, diagpos => ones(Nux_in))
    # Map from Npx+2 points to Nux_t-1 points (ux faces)
    Bkux = Bmap


    ## Y-direction

    # Gyi: integration and gyd: differentiation
    gyd = gy
    gyd[1] = hy[1]
    gyd[end] = hy[end]

    # Hyi: integration and hyd: differentiation
    # Map to find suitable size
    hyi = hy


    # Restrict Ny+2 to Nvy_in+1 points
    if bc.v.low == :dirichlet && bc.v.up == :dirichlet
        yin = y[2:end-1]
        hyd = hy
        gyi = gy[2:end-1]
        diagpos = 1

        if setup.discretization.order4
            hyd3 = [hy3[1]; hy3; hy3[end]]
            hyd13 = [hy[1]; hy; hy[end]]
            gyd3 = [2 * gy[1] + gy[2] + gy[3]; gy3; 2 * gy[end] + gy[end-1] + gy[end-2]]
            gyd13 = [gy[2]; 2 * gy[1]; gy[2:end-1]; 2 * gy[end]; gy[end-1]]
            gyi3 = gy3[2:end-1]
        end
    end

    if bc.v.low == :dirichlet && bc.v.up == :pressure
        yin = y[2:end]
        hyd = [hy; hy[end]]
        gyi = gy[2:end]
        diagpos = 1
    end

    if bc.v.low == :pressure && bc.v.up == :dirichlet
        yin = y[1:end-1]
        hyd = [hy[1]; hy]
        gyi = gy[1:end-1]
        diagpos = 0
    end

    if bc.v.low == :pressure && bc.v.up == :pressure
        yin = y[1:end]
        hyd = [hy[1]; hy; hy[end]]
        gyi = gy
        diagpos = 0
    end

    if bc.v.low == :periodic && bc.v.up == :periodic
        yin = y[1:end-1]
        hyd = [hy[end]; hy]
        gyi = [gy[1] + gy[end]; gy[2:end-1]]
        gyd[1] = (hy[1] + hy[end]) / 2
        gyd[end] = (hy[1] + hy[end]) / 2
        diagpos = 0

        if order4
            hyd3 = [hy3[end-1]; hy3[end]; hy3; hy3[1]]
            hyd13 = [hy[end-1]; hy[end]; hy; hy[1]]
            gyd3 = [gy3[end-1]; gy3; gy3[2]]
            gyd13 = [gy[end-1]; gy[1] + gy[end]; gy[2:end-1]; gy[end] + gy[1]; gy[2]]
            gyi3 = gy3[1:end-1]
        end
    end

    Bmap = spdiagm(Nvy_in + 1, Ny + 2, diagpos => ones(Nvy_in + 1))

    # Matrix to map from Nuy_t-1 to Nvy_in points
    # (used in interpolation, convection_diffusion)
    Buvy = spdiagm(Nvy_in, Nuy_t - 1, diagpos => ones(Nvy_in))
    # Map from Npy+2 points to Nvy_t-1 points (vy faces)
    Bkvy = Bmap

    ##
    # Volume (area) of pressure control volumes
    Ωp = kron(hyi, hxi)
    Ωp⁻¹ = 1 ./ Ωp
    # Volume (area) of u control volumes
    Ωu = kron(hyi, gxi)
    Ωu⁻¹ = 1 ./ Ωu
    # Volume of ux volumes
    Ωux = kron(hyi, hxd)
    # Volume of uy volumes
    Ωuy = kron(gyd, gxi)
    # Volume (area) of v control volumes
    Ωv = kron(gyi, hxi)
    Ωv⁻¹ = 1 ./ Ωv
    # Volume of vx volumes
    Ωvx = kron(gyi, gxd)
    # Volume of vy volumes
    Ωvy = kron(hyd, hxi)
    # Volume (area) of vorticity control volumes
    Ωvort = kron(gyi, gxi)
    Ωvort⁻¹ = 1 ./ Ωvort
\
    Ω = [Ωu; Ωv]
    Ω⁻¹ = [Ωu⁻¹; Ωv⁻¹]

    if setup.discretization.order4
        # Differencing for second order operators on the fourth order mesh
        Ωux1 = kron(hyi, hxd13)
        Ωuy1 = kron(gyd13, gxi)
        Ωvx1 = kron(gyi, gxd13)
        Ωvy1 = kron(hyd13, hxi)

        # Volume (area) of pressure control volumes
        Ωp3 = kron(hyi3, hxi3)
        # Volume (area) of u-vel control volumes
        Ωu3 = kron(hyi3, gxi3)
        # Volume (area) of v-vel control volumes
        Ωv3 = kron(gyi3, hxi3)
        # Volume (area) of dudx control volumes
        Ωux3 = kron(hyi3, hxd3)
        # Volume (area) of dudy control volumes
        Ωuy3 = kron(gyd3, gxi3)
        # Volume (area) of dvdx control volumes
        Ωvx3 = kron(gyi3, gxd3)
        # Volume (area) of dvdy control volumes
        Ωvy3 = kron(hyd3, hxi3)

        Ωu1 = Ωu
        Ωv1 = Ωv

        Ωu = α * Ωu1 - Ωu3
        Ωv = α * Ωv1 - Ωv3
        Ωu⁻¹ = 1 ./ Ωu
        Ωv⁻¹ = 1 ./ Ωv
        Ω = [Ωu; Ωv]
        Ω⁻¹ = [Ωu⁻¹; Ωv⁻¹]

        Ωux = @. α * Ωux1 - Ωux3
        Ωuy = @. α * Ωuy1 - Ωuy3
        Ωvx = @. α * Ωvx1 - Ωvx3
        Ωvy = @. α * Ωvy1 - Ωvy3

        Ωvort1 = Ωvort
        Ωvort3 = kron(gyi3, gxi3)
        # Ωvort = α*Ωvort - Ωvort3;
    end

    # Metrics that can be useful for initialization:
    xu = kron(ones(1, Nuy_in), xin)
    yu = kron(yp, ones(Nux_in))
    xu = reshape(xu, Nux_in, Nuy_in)
    yu = reshape(yu, Nux_in, Nuy_in)

    xv = kron(ones(1, Nvy_in), xp)
    yv = kron(yin, ones(Nvx_in))
    xv = reshape(xv, Nvx_in, Nvy_in)
    yv = reshape(yv, Nvx_in, Nvy_in)

    xpp = kron(ones(Ny), xp)
    ypp = kron(yp, ones(Nx))
    xpp = reshape(xpp, Nx, Ny)
    ypp = reshape(ypp, Nx, Ny)

    # Indices of unknowns in velocity vector
    indu = 1:Nu
    indv = Nu+1:Nu+Nv
    indV = 1:NV
    indp = NV+1:NV+Np

    ## Store quantities in the structure
    setup.grid.Npx = Npx
    setup.grid.Npy = Npy
    setup.grid.Np = Np

    setup.grid.Nux_in = Nux_in
    setup.grid.Nux_b = Nux_b
    setup.grid.Nux_t = Nux_t

    setup.grid.Nuy_in = Nuy_in
    setup.grid.Nuy_b = Nuy_b
    setup.grid.Nuy_t = Nuy_t

    setup.grid.Nvx_in = Nvx_in
    setup.grid.Nvx_b = Nvx_b
    setup.grid.Nvx_t = Nvx_t

    setup.grid.Nvy_in = Nvy_in
    setup.grid.Nvy_b = Nvy_b
    setup.grid.Nvy_t = Nvy_t

    setup.grid.Nu = Nu
    setup.grid.Nv = Nv
    setup.grid.NV = NV
    setup.grid.Ntot = Ntot

    setup.grid.N1 = N1
    setup.grid.N2 = N2
    setup.grid.N3 = N3
    setup.grid.N4 = N4

    setup.grid.Ωp = Ωp
    setup.grid.Ωp⁻¹ = Ωp⁻¹
    setup.grid.Ω = Ω
    setup.grid.Ωu = Ωu
    setup.grid.Ωv = Ωv
    setup.grid.Ω⁻¹ = Ω⁻¹
    setup.grid.Ωu⁻¹ = Ωu⁻¹
    setup.grid.Ωv⁻¹ = Ωv⁻¹
    setup.grid.Ωux = Ωux
    setup.grid.Ωvx = Ωvx
    setup.grid.Ωuy = Ωuy
    setup.grid.Ωvy = Ωvy
    setup.grid.Ωvort = Ωvort

    setup.grid.hxi = hxi
    setup.grid.hyi = hyi
    setup.grid.hxd = hxd
    setup.grid.hyd = hyd

    setup.grid.gxi = gxi
    setup.grid.gyi = gyi
    setup.grid.gxd = gxd
    setup.grid.gyd = gyd

    setup.grid.Buvy = Buvy
    setup.grid.Bvux = Bvux
    setup.grid.Bkux = Bkux
    setup.grid.Bkvy = Bkvy

    setup.grid.xin = xin
    setup.grid.yin = yin

    setup.grid.xu = xu
    setup.grid.yu = yu
    setup.grid.xv = xv
    setup.grid.yv = yv
    setup.grid.xpp = xpp
    setup.grid.ypp = ypp

    setup.grid.indu = indu
    setup.grid.indv = indv
    setup.grid.indV = indV
    setup.grid.indp = indp

    if order4
        setup.grid.hx3 = hx3
        setup.grid.hy3 = hy3
        setup.grid.hxi3 = hxi3
        setup.grid.hyi3 = hyi3
        setup.grid.gxi3 = gxi3
        setup.grid.gyi3 = gyi3
        setup.grid.hxd13 = hxd13
        setup.grid.hxd3 = hxd3
        setup.grid.hyd13 = hyd13
        setup.grid.hyd3 = hyd3
        setup.grid.gxd13 = gxd13
        setup.grid.gxd3 = gxd3
        setup.grid.gyd13 = gyd13
        setup.grid.gyd3 = gyd3
        setup.grid.Ωux1 = Ωux1
        setup.grid.Ωux3 = Ωux3
        setup.grid.Ωuy1 = Ωuy1
        setup.grid.Ωuy3 = Ωuy3
        setup.grid.Ωvx1 = Ωvx1
        setup.grid.Ωvx3 = Ωvx3
        setup.grid.Ωvy1 = Ωvy1
        setup.grid.Ωvy3 = Ωvy3
        setup.grid.Ωvort3 = Ωvort3
    end

    # Plot the grid: velocity points and pressure points
    if setup.visualization.plotgrid
        plot_staggered(x, y)
    end

    setup
end
