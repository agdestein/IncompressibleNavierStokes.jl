function operator_mesh!(setup)
    @unpack bc = setup
    @unpack Nx, Ny, Nz, x, y, z, hx, hy, hz, gx, gy, gz, xp, yp, zp = setup.grid

    # Number of pressure points
    Npx = Nx
    Npy = Ny
    Npz = Nz
    Np = Npx * Npy * Npz

    ## u-volumes
    # x[1]   x[2]   x[3] ....      x[Nx]   x[Nx+1]
    # |      |      |              |       |
    # |      |      |              |       |
    # Dirichlet BC:
    # ULe    u[1]   u[2] ....      u(Nx-1) uRi
    # Periodic BC:
    # u[1]   u[2]   u[3] ....      u[Nx]   u[1]
    # Pressure BC:
    # u[1]   u[2]   u[3] ....      u[Nx]   u[Nx+1]

    # x-dir
    Nux_b = 2               # Boundary points
    Nux_in = Nx + 1         # Inner points
    Nux_in -= bc.u.x[1] ∈ [:dirichlet, :symmetric]
    Nux_in -= bc.u.x[2] ∈ [:dirichlet, :symmetric]
    Nux_in -= bc.u.x == (:periodic, :periodic)
    Nux_t = Nux_in + Nux_b  # Total number

    # y-dir
    Nuy_b = 2               # Boundary points
    Nuy_in = Ny             # Inner points
    Nuy_t = Nuy_in + Nuy_b  # Total number
    
    # z-dir
    Nuz_b = 2               # Boundary points
    Nuz_in = Nz             # Inner points
    Nuz_t = Nuz_in + Nuz_b  # Total number
    
    # Total number
    Nu = Nux_in * Nuy_in * Nuz_in


    ## v-volumes

    # X-dir
    Nvx_b = 2               # Boundary points
    Nvx_in = Nx             # Inner points
    Nvx_t = Nvx_in + Nvx_b  # Total number

    # Y-dir
    Nvy_b = 2               # Boundary points
    Nvy_in = Ny + 1         # Inner points
    Nvy_in -= bc.v.y[1] ∈ [:dirichlet, :symmetric]
    Nvy_in -= bc.v.y[2] ∈ [:dirichlet, :symmetric]
    Nvy_in -= bc.v.y == (:periodic, :periodic)
    Nvy_t = Nvy_in + Nvy_b # Total number

    # z-dir
    Nvz_b = 2               # Boundary points
    Nvz_in = Nz             # Inner points
    Nvz_t = Nvz_in + Nvz_b  # Total number
    
    # Total number
    Nv = Nvx_in * Nvy_in * Nvz_in


    ## w-volumes

    # X-dir
    Nwx_b = 2               # Boundary points
    Nwx_in = Nx             # Inner points
    Nwx_t = Nwx_in + Nwx_b  # Total number

    # Y-dir
    Nwy_b = 2               # Boundary points
    Nwy_in = Ny             # Inner points
    Nwy_t = Nwy_in + Nwy_b  # Total number
    
    # z-dir
    Nwz_b = 2               # Boundary points
    Nwz_in = Ny + 1         # Inner points
    Nwz_in -= bc.w.z[1] ∈ [:dirichlet, :symmetric]
    Nwz_in -= bc.w.z[2] ∈ [:dirichlet, :symmetric]
    Nwz_in -= bc.w.z == (:periodic, :periodic)
    Nwz_t = Nwz_in + Nwz_b  # Total number

    # Total number
    Nw = Nwx_in * Nwy_in * Nwz_in


    # Total number of velocity points
    NV = Nu + Nv + Nw


    ## Adapt mesh metrics depending on number of volumes

    ## X-direction

    # gxd: differentiation
    gxd = copy(gx)
    gxd[1] = hx[1]
    gxd[end] = hx[end]

    # hxi: integration and hxd: differentiation
    # Map to find suitable size
    hxi = copy(hx)

    # Restrict Nx+2 to Nux_in+1 points
    if bc.u.x == (:dirichlet, :dirichlet)
        xin = x[2:end-1]
        hxd = copy(hx)
        gxi = gx[2:end-1]
        diagpos = 1
    elseif bc.u.x == (:dirichlet, :pressure)
        xin = x[2:end]
        hxd = [hx; hx[end]]
        gxi = gx[2:end]
        diagpos = 1
    elseif bc.u.x == (:pressure, :dirichlet)
        xin = x[1:end-1]
        hxd = [hx[1]; hx]
        gxi = gx[1:end-1]
        diagpos = 0
    elseif bc.u.x == (:pressure, :pressure)
        xin = x[1:end]
        hxd = [hx[1]; hx; hx[end]]
        gxi = copy(gx)
        diagpos = 0
    elseif bc.u.x == (:periodic, :periodic)
        xin = x[1:end-1]
        hxd = [hx[end]; hx]
        gxi = [gx[1] + gx[end]; gx[2:end-1]]
        gxd[1] = (hx[1] + hx[end]) / 2
        gxd[end] = (hx[1] + hx[end]) / 2
        diagpos = 0
    end

    Bmap = spdiagm(Nux_in + 1, Nx + 2, diagpos => ones(Nux_in + 1))

    # Matrix to map from Nvx_t-1 to Nux_in points
    # (used in interpolation, convection_diffusion, viscosity)
    Bvux = spdiagm(Nux_in, Nvx_t - 1, diagpos => ones(Nux_in))
    Bwux = spdiagm(Nux_in, Nwx_t - 1, diagpos => ones(Nux_in))
    # Map from Npx+2 points to Nux_t-1 points (ux faces)
    Bkux = copy(Bmap)


    ## Y-direction

    # Gyi: integration and gyd: differentiation
    gyd = copy(gy)
    gyd[1] = hy[1]
    gyd[end] = hy[end]

    # Hyi: integration and hyd: differentiation
    # Map to find suitable size
    hyi = copy(hy)

    # Restrict Ny+2 to Nvy_in+1 points
    if bc.v.y == (:dirichlet, :dirichlet)
        yin = y[2:end-1]
        hyd = copy(hy)
        gyi = gy[2:end-1]
        diagpos = 1
    elseif bc.v.y == (:dirichlet, :pressure)
        yin = y[2:end]
        hyd = [hy; hy[end]]
        gyi = gy[2:end]
        diagpos = 1
    elseif bc.v.y == (:pressure, :dirichlet)
        yin = y[1:end-1]
        hyd = [hy[1]; hy]
        gyi = gy[1:end-1]
        diagpos = 0
    elseif bc.v.y == (:pressure, :pressure)
        yin = y[1:end]
        hyd = [hy[1]; hy; hy[end]]
        gyi = copy(gy)
        diagpos = 0
    elseif bc.v.y == (:periodic, :periodic)
        yin = y[1:end-1]
        hyd = [hy[end]; hy]
        gyi = [gy[1] + gy[end]; gy[2:end-1]]
        gyd[1] = (hy[1] + hy[end]) / 2
        gyd[end] = (hy[1] + hy[end]) / 2
        diagpos = 0
    end

    Bmap = spdiagm(Nvy_in + 1, Ny + 2, diagpos => ones(Nvy_in + 1))

    # Matrix to map from Nuy_t-1 to Nvy_in points
    # (used in interpolation, convection_diffusion)
    Buvy = spdiagm(Nvy_in, Nuy_t - 1, diagpos => ones(Nvy_in))
    Bwvy = spdiagm(Nvy_in, Nwy_t - 1, diagpos => ones(Nvy_in))
    # Map from Npy+2 points to Nvy_t-1 points (vy faces)
    Bkvy = copy(Bmap)


    ## Z-direction

    # gzi: integration and gzd: differentiation
    gzd = copy(gz)
    gzd[1] = hz[1]
    gzd[end] = hz[end]

    # hzi: integration and hzd: differentiation
    # Map to find suitable size
    hzi = copy(hz)

    # Restrict Nz+2 to Nvz_in+1 points
    if bc.w.z == (:dirichlet, :dirichlet)
        zin = z[2:end-1]
        hzd = copy(hz)
        gzi = gz[2:end-1]
        diagpos = 1
    elseif bc.w.z == (:dirichlet, :pressure)
        zin = z[2:end]
        hzd = [hz; hz[end]]
        gzi = gz[2:end]
        diagpos = 1
    elseif bc.w.z == (:pressure, :dirichlet)
        zin = z[1:end-1]
        hzd = [hz[1]; hz]
        gzi = gz[1:end-1]
        diagpos = 0
    elseif bc.w.z == (:pressure, :pressure)
        zin = z[1:end]
        hzd = [hz[1]; hz; hz[end]]
        gzi = copy(gz)
        diagpos = 0
    elseif bc.w.z == (:periodic, :periodic)
        zin = z[1:end-1]
        hzd = [hz[end]; hz]
        gzi = [gz[1] + gz[end]; gz[2:end-1]]
        gzd[1] = (hz[1] + hz[end]) / 2
        gzd[end] = (hz[1] + hz[end]) / 2
        diagpos = 0
    end

    Bmap = spdiagm(Nwz_in + 1, Nz + 2, diagpos => ones(Nwz_in + 1))

    # Matrix to map from Nuz_t-1 to Nvz_in points
    # (used in interpolation, convection_diffusion)
    Buwz = spdiagm(Nwz_in, Nuz_t - 1, diagpos => ones(Nwz_in))
    Bvwz = spdiagm(Nwz_in, Nvz_t - 1, diagpos => ones(Nwz_in))
    # Map from Npy+2 points to Nvy_t-1 points (vy faces)
    Bkwz = copy(Bmap)


    ## Volumes
    # Volume (area) of pressure control volumes
    Ωp = hzi ⊗ hyi ⊗ hxi
    Ωp⁻¹ = 1 ./ Ωp

    # Volume (area) of u control volumes
    Ωu = hzi ⊗ hyi ⊗ gxi
    Ωu⁻¹ = 1 ./ Ωu

    # Volume (area) of v control volumes
    Ωv = hzi ⊗ gyi ⊗ hxi
    Ωv⁻¹ = 1 ./ Ωv

    # Volume (area) of w control volumes
    Ωw = gzi ⊗ hyi ⊗ hxi
    Ωw⁻¹ = 1 ./ Ωw

    Ω = [Ωu; Ωv; Ωw]
    Ω⁻¹ = [Ωu⁻¹; Ωv⁻¹; Ωw⁻¹]

    # Metrics that can be useful for initialization:
    xu = ones(Nuz_in) ⊗ ones(Nuy_in) ⊗ xin
    yu = ones(Nuz_in) ⊗ yp ⊗ ones(Nux_in)
    zu = zp ⊗ ones(Nuy_in) ⊗ ones(Nux_in)
    xu = reshape(xu, Nux_in, Nuy_in, Nuz_in)
    yu = reshape(yu, Nux_in, Nuy_in, Nuz_in)
    zu = reshape(zu, Nux_in, Nuy_in, Nuz_in)

    xv = ones(Nvz_in) ⊗ ones(Nvy_in) ⊗ xp
    yv = ones(Nvz_in) ⊗ yin ⊗ ones(Nvx_in)
    zv = zp ⊗ ones(Nvy_in) ⊗ ones(Nvx_in)
    xv = reshape(xv, Nvx_in, Nvy_in, Nvz_in)
    yv = reshape(yv, Nvx_in, Nvy_in, Nvz_in)
    zv = reshape(zv, Nvx_in, Nvy_in, Nvz_in)

    xw = ones(Nwz_in) ⊗ ones(Nwy_in) ⊗ xp
    yw = ones(Nwz_in) ⊗ yp ⊗ ones(Nwx_in)
    zw = zin ⊗ ones(Nwy_in) ⊗ ones(Nwx_in)
    xw = reshape(xw, Nwx_in, Nwy_in, Nwz_in)
    yw = reshape(yw, Nwx_in, Nwy_in, Nwz_in)
    zw = reshape(zw, Nwx_in, Nwy_in, Nwz_in)

    xpp = ones(Nz) ⊗ ones(Ny) ⊗ xp
    ypp = ones(Nz) ⊗ yp ⊗ ones(Nx)
    zpp = zp ⊗ ones(Ny) ⊗ ones(Nx)
    xpp = reshape(xpp, Nx, Ny, Nz)
    ypp = reshape(ypp, Nx, Ny, Nz)
    zpp = reshape(zpp, Nx, Ny, Nz)

    # Indices of unknowns in velocity vector
    indu = 1:Nu
    indv = Nu .+ (1:Nv)
    indw = Nu + Nv .+ (1:Nw)
    indV = 1:NV
    indp = NV+1:NV+Np

    ## Store quantities in the structure
    @pack! setup.grid = Npx, Npy, Npz, Np
    @pack! setup.grid = Nux_in, Nux_b, Nux_t
    @pack! setup.grid = Nuy_in, Nuy_b, Nuy_t
    @pack! setup.grid = Nuz_in, Nuz_b, Nuz_t
    @pack! setup.grid = Nvx_in, Nvx_b, Nvx_t
    @pack! setup.grid = Nvy_in, Nvy_b, Nvy_t
    @pack! setup.grid = Nvz_in, Nvz_b, Nvz_t
    @pack! setup.grid = Nu, Nv, Nw, NV
    @pack! setup.grid = Ωp, Ωp⁻¹, Ω, Ω⁻¹, Ωu, Ωv, Ωw, Ωu⁻¹, Ωv⁻¹, Ωw⁻¹
    @pack! setup.grid = hxi, hyi, hzi, hxd, hyd, hzd
    @pack! setup.grid = gxi, gyi, gzi, gxd, gyd, gzd
    @pack! setup.grid = Buvy, Bvux, Buwz, Bwux, Bvwz, Bwvy, Bkux, Bkvy, Bkwz
    @pack! setup.grid = xin, yin, zin
    @pack! setup.grid = xu, yu, zu, xv, yv, zv, xw, yw, zw, xpp, ypp, zpp
    @pack! setup.grid = indu, indv, indw, indV, indp

    setup
end
