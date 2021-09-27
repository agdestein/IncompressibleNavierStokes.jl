function operator_mesh!(setup)
    BC = setup.BC
    order4 = setup.discretization.order4
    α = setup.discretization.α

    ## pressure volumes
    @unpack Nx, Ny, x, y, hx, hy, gx, gy, xp, yp = setup.grid
    # @unpack gx, gy = setup.grid

    # number of pressure points
    Npx = Nx
    Npy = Ny
    Np = Npx * Npy

    ## u-volumes
    # x[1]   x[2]   x[3] ....      x[Nx]   x[Nx+1]
    # |      |      |              |       |
    # |      |      |              |       |
    # Dirichlet BC:
    # uLe    u[1]   u[2] ....      u(Nx-1) uRi
    # periodic BC:
    # u[1]   u[2]   u[3] ....      u[Nx]   u[1]
    # pressure BC:
    # u[1]   u[2]   u[3] ....      u[Nx]   u[Nx+1]

    # x-dir
    Nux_b = 2               # boundary points
    Nux_in = Nx + 1            # inner points
    if BC.u.left == "dir" || BC.u.left == "sym"
        Nux_in = Nux_in - 1
    end
    if BC.u.right == "dir" || BC.u.right == "sym"
        Nux_in = Nux_in - 1
    end
    if BC.u.left == "per" && BC.u.right == "per"
        Nux_in = Nux_in - 1
    end
    Nux_t = Nux_in + Nux_b  # total number

    # y-dir
    Nuy_b = 2               # boundary points
    Nuy_in = Ny              # inner points
    Nuy_t = Nuy_in + Nuy_b  # total number

    # total number
    Nu = Nux_in * Nuy_in


    ## v-volumes

    # x-dir
    Nvx_b = 2               # boundary points
    Nvx_in = Nx              # inner points
    Nvx_t = Nvx_in + Nvx_b  # total number

    # y-dir
    Nvy_b = 2               # boundary points
    Nvy_in = Ny + 1            # inner points
    if BC.v.low == "dir" || BC.v.low == "sym"
        Nvy_in = Nvy_in - 1
    end
    if BC.v.up == "dir" || BC.v.up == "sym"
        Nvy_in = Nvy_in - 1
    end
    if BC.v.low == "per" && BC.v.up == "per"
        Nvy_in = Nvy_in - 1
    end
    Nvy_t = Nvy_in + Nvy_b  # total number

    # total number
    Nv = Nvx_in * Nvy_in

    # total number of velocity points
    NV = Nu + Nv

    # total number of unknowns
    Ntot = NV + Np

    ## extra variables
    N1 = (Nux_in + 1) * Nuy_in #size(Iu_ux, 1);
    N2 = Nux_in * (Nuy_in + 1) #size(Iv_uy, 1);
    N3 = (Nvx_in + 1) * Nvy_in # size(Iu_vx, 1);
    N4 = Nvx_in * (Nvy_in + 1) # size(Iv_vy, 1);


    ## for a grid with three times larger volumes:
    if order4
        hx3 = zeros(Nx, 1)
        hx3[2:end-1] = hx[1:end-2] + hx[2:end-1] + hx[3:end]
        if BC.u.left == "per" && BC.u.right == "per"
            hx3[1] = hx[end] + hx[1] + hx[2]
            hx3[end] = hx[end-1] + hx[end] + hx[1]
        else
            hx3[1] = 2 * hx[1] + hx[2]
            hx3[end] = hx[end-1] + 2 * hx[end]
        end

        hy3 = zeros(Ny, 1)
        hy3[2:end-1] = hy[1:end-2] + hy[2:end-1] + hy[3:end]
        if BC.v.low == "per" && BC.v.up == "per"
            hy3[1] = hy[end] + hy[1] + hy[2]
            hy3[end] = hy[end-1] + hy[end] + hy[1]
        else
            hy3[1] = 2 * hy[1] + hy[2]
            hy3[end] = hy[end-1] + 2 * hy[end]
        end

        hxi3 = hx3
        hyi3 = hy3


        # distance between pressure points
        gx3 = zeros(Nx + 1, 1)
        gx3[3:Nx-1] = gx[2:end-3] + gx[3:end-2] + gx[4:end-1]
        if BC.u.left == "per" && BC.u.right == "per"
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

        # distance between pressure points
        gy3 = zeros(Ny + 1, 1)
        gy3[3:Ny-1] = gy[2:end-3] + gy[3:end-2] + gy[4:end-1]
        if BC.v.low == "per" && BC.v.up == "per"
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

    ## adapt mesh metrics depending on number of volumes

    ## x-direction

    # gxd: differentiation
    gxd = gx
    gxd[1] = hx[1]
    gxd[end] = hx[end]

    # hxi: integration and hxd: differentiation
    # map to find suitable size
    hxi = hx

    # restrict Nx+2 to Nux_in+1 points
    if BC.u.left == "dir" && BC.u.right == "dir"
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

    if BC.u.left == "dir" && BC.u.right == "pres"
        xin = x[2:end]
        hxd = [hx; hx[end]]
        gxi = gx[2:end]
        diagpos = 1
    end

    if BC.u.left == "pres" && BC.u.right == "dir"
        xin = x[1:end-1]
        hxd = [hx[1]; hx]
        gxi = gx[1:end-1]
        diagpos = 0
    end

    if BC.u.left == "pres" && BC.u.right == "pres"
        xin = x[1:end]
        hxd = [hx[1]; hx; hx[end]]
        gxi = gx
        diagpos = 0
    end

    if BC.u.left == "per" && BC.u.right == "per"
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

    # matrix to map from Nvx_t-1 to Nux_in points
    # (used in interpolation, convection_diffusion, viscosity)
    Bvux = spdiagm(Nux_in, Nvx_t - 1, diagpos => ones(Nvx_t - 1))
    # map from Npx+2 points to Nux_t-1 points (ux faces)
    Bkux = Bmap


    ## y-direction

    # gyi: integration and gyd: differentiation
    gyd = gy
    gyd[1] = hy[1]
    gyd[end] = hy[end]

    # hyi: integration and hyd: differentiation
    # map to find suitable size
    hyi = hy


    # restrict Ny+2 to Nvy_in+1 points
    if BC.v.low == "dir" && BC.v.up == "dir"
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

    if BC.v.low == "dir" && BC.v.up == "pres"
        yin = y[2:end]
        hyd = [hy; hy[end]]
        gyi = gy[2:end]
        diagpos = 1
    end

    if BC.v.low == "pres" && BC.v.up == "dir"
        yin = y[1:end-1]
        hyd = [hy[1]; hy]
        gyi = gy[1:end-1]
        diagpos = 0
    end

    if BC.v.low == "pres" && BC.v.up == "pres"
        yin = y[1:end]
        hyd = [hy[1]; hy; hy[end]]
        gyi = gy
        diagpos = 0
    end

    if BC.v.low == "per" && BC.v.up == "per"
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

    # matrix to map from Nuy_t-1 to Nvy_in points
    # (used in interpolation, convection_diffusion)
    Buvy = spdiagm(Nvy_in, Nuy_t - 1, diagpos => ones(Nuy_t - 1, 1))
    # map from Npy+2 points to Nvy_t-1 points (vy faces)
    Bkvy = Bmap

    ##
    # volume (area) of pressure control volumes
    Omp = kron(hyi, hxi)
    Omp_inv = 1 ./ Omp
    # volume (area) of u control volumes
    Omu = kron(hyi, gxi)
    Omu_inv = 1 ./ Omu
    # volume of ux volumes
    Omux = kron(hyi, hxd)
    # volume of uy volumes
    Omuy = kron(gyd, gxi)
    # volume (area) of v control volumes
    Omv = kron(gyi, hxi)
    Omv_inv = 1 ./ Omv
    # volume of vx volumes
    Omvx = kron(gyi, gxd)
    # volume of vy volumes
    Omvy = kron(hyd, hxi)
    # volume (area) of vorticity control volumes
    Omvort = kron(gyi, gxi)
    Omvort_inv = 1 ./ Omvort

    Om = [Omu; Omv]
    Om_inv = [Omu_inv; Omv_inv]

    if setup.discretization.order4
        # differencing for second order operators on the fourth order mesh
        Omux1 = kron(hyi, hxd13)
        Omuy1 = kron(gyd13, gxi)
        Omvx1 = kron(gyi, gxd13)
        Omvy1 = kron(hyd13, hxi)

        # volume (area) of pressure control volumes
        Omp3 = kron(hyi3, hxi3)
        # volume (area) of u-vel control volumes
        Omu3 = kron(hyi3, gxi3)
        # volume (area) of v-vel control volumes
        Omv3 = kron(gyi3, hxi3)
        # volume (area) of dudx control volumes
        Omux3 = kron(hyi3, hxd3)
        # volume (area) of dudy control volumes
        Omuy3 = kron(gyd3, gxi3)
        # volume (area) of dvdx control volumes
        Omvx3 = kron(gyi3, gxd3)
        # volume (area) of dvdy control volumes
        Omvy3 = kron(hyd3, hxi3)

        Omu1 = Omu
        Omv1 = Omv

        Omu = α * Omu1 - Omu3
        Omv = α * Omv1 - Omv3
        Omu_inv = 1 ./ Omu
        Omv_inv = 1 ./ Omv
        Om = [Omu; Omv]
        Om_inv = [Omu_inv; Omv_inv]

        Omux = α * Omux1 - Omux3
        Omuy = α * Omuy1 - Omuy3
        Omvx = α * Omvx1 - Omvx3
        Omvy = α * Omvy1 - Omvy3

        Omvort1 = Omvort
        Omvort3 = kron(gyi3, gxi3)
        #     Omvort = α*Omvort - Omvort3;
    end

    # metrics that can be useful for initialization:
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

    # indices of unknowns in velocity vector
    indu = (1:Nu)'
    indv = (Nu+1:Nu+Nv)'
    indV = [indu; indv]
    indp = (NV+1:NV+Np)'

    ## store quantities in the structure
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

    setup.grid.Omp = Omp
    setup.grid.Omp_inv = Omp_inv
    setup.grid.Om = Om
    setup.grid.Omu = Omu
    setup.grid.Omv = Omv
    setup.grid.Om_inv = Om_inv
    setup.grid.Omu_inv = Omu_inv
    setup.grid.Omv_inv = Omv_inv
    setup.grid.Omux = Omux
    setup.grid.Omvx = Omvx
    setup.grid.Omuy = Omuy
    setup.grid.Omvy = Omvy
    setup.grid.Omvort = Omvort

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
        setup.grid.Omux1 = Omux1
        setup.grid.Omux3 = Omux3
        setup.grid.Omuy1 = Omuy1
        setup.grid.Omuy3 = Omuy3
        setup.grid.Omvx1 = Omvx1
        setup.grid.Omvx3 = Omvx3
        setup.grid.Omvy1 = Omvy1
        setup.grid.Omvy3 = Omvy3
        setup.grid.Omvort3 = Omvort3
    end

    # plot the grid: velocity points and pressure points
    if setup.visualization.plotgrid
        plot_staggered(x, y)
    end

    setup
end
