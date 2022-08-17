"""
    create_grid(x, y; T = eltype(x))

Create nonuniform cartesian box mesh `x` × `y`.
"""
function create_grid(x, y; T = eltype(x))
    Nx = length(x) - 1
    Ny = length(y) - 1
    xlims = (x[1], x[end])
    ylims = (y[1], y[end])

    # Pressure positions
    xp = (x[1:(end-1)] + x[2:end]) / 2
    yp = (y[1:(end-1)] + y[2:end]) / 2

    # Distance between velocity points
    hx = diff(x)
    hy = diff(y)

    # Distance between pressure points
    gx = zeros(Nx + 1)
    gx[1] = hx[1] / 2
    gx[2:Nx] = (hx[1:(Nx-1)] + hx[2:Nx]) / 2
    gx[Nx+1] = hx[end] / 2

    gy = zeros(Ny + 1)
    gy[1] = hy[1] / 2
    gy[2:Ny] = (hy[1:(Ny-1)] + hy[2:Ny]) / 2
    gy[Ny+1] = hy[end] / 2

    # Number of pressure points
    Npx = Nx
    Npy = Ny
    Np = Npx * Npy

    ## u-volumes
    # Periodic BC:
    # x[1]  x[2]  x[3] ... x[Nx]  x[Nx+1]
    #  |     |     |         |      |
    # u[1]  u[2]  u[3] ... u[Nx]  u[1]

    # x-dir
    Nux_b = 2               # Boundary points
    Nux_in = Nx             # Inner points
    Nux_t = Nux_in + Nux_b  # Total number

    # Y-dir
    Nuy_b = 2               # Boundary points
    Nuy_in = Ny             # Inner points
    Nuy_t = Nuy_in + Nuy_b  # Total number

    # Total number
    Nu = Nux_in * Nuy_in


    ## v-volumes

    # X-dir
    Nvx_b = 2               # Boundary points
    Nvx_in = Nx             # Inner points
    Nvx_t = Nvx_in + Nvx_b  # Total number

    # Y-dir
    Nvy_b = 2               # Boundary points
    Nvy_in = Ny             # Inner points
    Nvy_t = Nvy_in + Nvy_b  # Total number

    # Total number
    Nv = Nvx_in * Nvy_in

    # Total number of velocity points
    NV = Nu + Nv

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
    xin = x[1:end-1]
    hxd = [hx[end]; hx]
    gxi = [gx[1] + gx[end]; gx[2:end-1]]
    gxd[1] = (hx[1] + hx[end]) / 2
    gxd[end] = (hx[1] + hx[end]) / 2
    diagpos = 0

    Bmap = spdiagm(Nux_in + 1, Nx + 2, diagpos => ones(Nux_in + 1))

    # Matrix to map from Nvx_t-1 to Nux_in points
    # (used in interpolation, convection_diffusion, viscosity)
    Bvux = spdiagm(Nux_in, Nvx_t - 1, diagpos => ones(Nux_in))

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
    yin = y[1:end-1]
    hyd = [hy[end]; hy]
    gyi = [gy[1] + gy[end]; gy[2:end-1]]
    gyd[1] = (hy[1] + hy[end]) / 2
    gyd[end] = (hy[1] + hy[end]) / 2
    diagpos = 0

    Bmap = spdiagm(Nvy_in + 1, Ny + 2, diagpos => ones(Nvy_in + 1))

    # Matrix to map from Nuy_t-1 to Nvy_in points
    # (used in interpolation, convection_diffusion)
    Buvy = spdiagm(Nvy_in, Nuy_t - 1, diagpos => ones(Nvy_in))

    # Map from Npy+2 points to Nvy_t-1 points (vy faces)
    Bkvy = copy(Bmap)

    ##
    # Volume (area) of pressure control volumes
    Ωp = hyi ⊗ hxi

    # Volume (area) of u control volumes
    Ωu = hyi ⊗ gxi

    # Volume of ux volumes
    Ωux = hyi ⊗ hxd

    # Volume of uy volumes
    Ωuy = gyd ⊗ gxi

    # Volume (area) of v control volumes
    Ωv = gyi ⊗ hxi

    # Volume of vx volumes
    Ωvx = gyi ⊗ gxd

    # Volume of vy volumes
    Ωvy = hyd ⊗ hxi

    Ω = [Ωu; Ωv]
    Ω⁻¹ = 1 ./ Ω

    # Metrics that can be useful for initialization:
    xu = ones(1, Nuy_in) ⊗ xin
    yu = yp ⊗ ones(Nux_in)
    xu = reshape(xu, Nux_in, Nuy_in)
    yu = reshape(yu, Nux_in, Nuy_in)

    xv = ones(1, Nvy_in) ⊗ xp
    yv = yin ⊗ ones(Nvx_in)
    xv = reshape(xv, Nvx_in, Nvy_in)
    yv = reshape(yv, Nvx_in, Nvy_in)

    xpp = ones(Ny) ⊗ xp
    ypp = yp ⊗ ones(Nx)
    xpp = reshape(xpp, Nx, Ny)
    ypp = reshape(ypp, Nx, Ny)

    # Indices of unknowns in velocity vector
    indu = 1:Nu
    indv = Nu .+ (1:Nv)
    indV = 1:NV
    indp = NV .+ (1:Np)

    Grid{T,2}(;
        Nx,
        Ny,
        xlims,
        ylims,
        x,
        y,
        xp,
        yp,
        hx,
        hy,
        gx,
        gy,
        Npx,
        Npy,
        Np,
        Nux_in,
        Nux_b,
        Nux_t,
        Nuy_in,
        Nuy_b,
        Nuy_t,
        Nvx_in,
        Nvx_b,
        Nvx_t,
        Nvy_in,
        Nvy_b,
        Nvy_t,
        Nu,
        Nv,
        NV,
        Ωp,
        Ω,
        Ω⁻¹,
        Ωux,
        Ωvx,
        Ωuy,
        Ωvy,
        hxi,
        hyi,
        hxd,
        hyd,
        gxi,
        gyi,
        gxd,
        gyd,
        Buvy,
        Bvux,
        Bkux,
        Bkvy,
        xin,
        yin,
        xu,
        yu,
        xv,
        yv,
        xpp,
        ypp,
        indu,
        indv,
        indV,
        indp,
    )
end

"""
    create_grid(x, y, z; T = eltype(x))

Create nonuniform cartesian box mesh `xlims` × `ylims` × `zlims`.
"""
function create_grid(x, y, z; T = eltype(x))
    Nx = length(x) - 1
    Ny = length(y) - 1
    Nz = length(z) - 1
    xlims = (x[1], x[end])
    ylims = (y[1], y[end])
    zlims = (z[1], z[end])

    # Pressure positions
    xp = (x[1:(end-1)] + x[2:end]) / 2
    yp = (y[1:(end-1)] + y[2:end]) / 2
    zp = (z[1:(end-1)] + z[2:end]) / 2

    # Distance between velocity points
    hx = diff(x)
    hy = diff(y)
    hz = diff(z)

    # Distance between pressure points
    gx = zeros(Nx + 1)
    gx[1] = hx[1] / 2
    gx[2:Nx] = (hx[1:(Nx-1)] + hx[2:Nx]) / 2
    gx[Nx+1] = hx[end] / 2

    gy = zeros(Ny + 1)
    gy[1] = hy[1] / 2
    gy[2:Ny] = (hy[1:(Ny-1)] + hy[2:Ny]) / 2
    gy[Ny+1] = hy[end] / 2

    gz = zeros(Nz + 1)
    gz[1] = hz[1] / 2
    gz[2:Nz] = (hz[1:(Nz-1)] + hz[2:Nz]) / 2
    gz[Nz+1] = hz[end] / 2

    # Number of pressure points
    Npx = Nx
    Npy = Ny
    Npz = Nz
    Np = Npx * Npy * Npz

    ## u-volumes
    # Periodic BC:
    # x[1]  x[2]  x[3] ... x[Nx]  x[Nx+1]
    #  |     |     |         |      |
    # u[1]  u[2]  u[3] ... u[Nx]  u[1]

    # x-dir
    Nux_b = 2               # Boundary points
    Nux_in = Nx             # Inner points
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

    # x-dir
    Nvx_b = 2               # Boundary points
    Nvx_in = Nx             # Inner points
    Nvx_t = Nvx_in + Nvx_b  # Total number

    # y-dir
    Nvy_b = 2               # Boundary points
    Nvy_in = Ny             # Inner points
    Nvy_t = Nvy_in + Nvy_b  # Total number

    # z-dir
    Nvz_b = 2               # Boundary points
    Nvz_in = Nz             # Inner points
    Nvz_t = Nvz_in + Nvz_b  # Total number

    # Total number
    Nv = Nvx_in * Nvy_in * Nvz_in


    ## w-volumes

    # x-dir
    Nwx_b = 2               # Boundary points
    Nwx_in = Nx             # Inner points
    Nwx_t = Nwx_in + Nwx_b  # Total number

    # y-dir
    Nwy_b = 2               # Boundary points
    Nwy_in = Ny             # Inner points
    Nwy_t = Nwy_in + Nwy_b  # Total number

    # z-dir
    Nwz_b = 2               # Boundary points
    Nwz_in = Nz             # Inner points
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
    hxd = [hx[1]; hx; hx[end]]

    # Restrict Nx+2 to Nux_in+1 points
    diagpos = 0

    Bmap = spdiagm(Nux_in + 1, Nx + 2, diagpos => ones(Nux_in + 1))
    Bmap_x_xin = spdiagm(Nux_in, Nx + 1, diagpos => ones(Nux_in))
    hxd = Bmap * hxd
    gxi = Bmap_x_xin * gx
    xin = Bmap_x_xin * x

    gxd[1] = (hx[1] + hx[end]) / 2
    gxd[end] = gxd[1]
    gxi[1] = gxd[1]
    hxd[1] = hx[end]

    # Matrix to map from Nvx_t-1 to Nux_in points
    Bvux = spdiagm(Nux_in, Nvx_t - 1, diagpos => ones(Nux_in))

    # Matrix to map from Nwx_t-1 to Nux_in points
    Bwux = spdiagm(Nux_in, Nwx_t - 1, diagpos => ones(Nux_in))

    # Map from Npx+2 points to Nux_t-1 points (ux faces)
    Bkux = copy(Bmap)


    ## y-direction

    # gyi: integration and gyd: differentiation
    gyd = copy(gy)
    gyd[1] = hy[1]
    gyd[end] = hy[end]

    # hyi: integration and hyd: differentiation
    # Map to find suitable size
    hyi = copy(hy)
    hyd = [hy[1]; hy; hy[end]]

    # Restrict Ny+2 to Nvy_in+1 points
    diagpos = 0

    Bmap = spdiagm(Nvy_in + 1, Ny + 2, diagpos => ones(Nvy_in + 1))
    Bmap_y_yin = spdiagm(Nvy_in, Ny + 1, diagpos => ones(Nvy_in))

    hyd = Bmap * hyd
    gyi = Bmap_y_yin * gy
    yin = Bmap_y_yin * y

    gyd[1] = (hy[1] + hy[end]) / 2
    gyd[end] = gyd[1]
    gyi[1] = gyd[1]
    hyd[1] = hy[end]

    # Matrix to map from Nuy_t-1 to Nvy_in points
    Buvy = spdiagm(Nvy_in, Nuy_t - 1, diagpos => ones(Nvy_in))

    # matrix to map from Nwy_t-1 to Nvy_in points
    Bwvy = spdiagm(Nvy_in, Nwy_t - 1, diagpos => ones(Nvy_in))

    # Map from Npy+2 points to Nvy_t-1 points (vy faces)
    Bkvy = copy(Bmap)


    ## z-direction

    # gzi: integration and gzd: differentiation
    gzd = copy(gz)
    gzd[1] = hz[1]
    gzd[end] = hz[end]

    # hzi: integration and hzd: differentiation
    # Map to find suitable size
    hzi = copy(hz)
    hzd = [hz[1]; hz; hz[end]]

    # Restrict Nz+2 to Nvz_in+1 points
    diagpos = 0

    shape = (Nwz_in + 1, Nz + 2)
    Bmap = spdiagm(shape..., diagpos => ones(Nwz_in + 1))
    Bmap_z_zin = spdiagm(Nwz_in, Nz + 1, diagpos => ones(Nwz_in))
    hzd = Bmap * hzd
    gzi = Bmap_z_zin * gz
    zin = Bmap_z_zin * z

    gzd[1] = (hz[1] + hz[end]) / 2
    gzd[end] = gzd[1]
    gzi[1] = gzd[1]
    hzd[1] = hz[end]

    # Matrix to map from Nuz_t-1 to Nwz_in points
    Buwz = spdiagm(Nwz_in, Nuz_t - 1, diagpos => ones(Nwz_in))

    # Matrix to map from Nvz_t-1 to Nwz_in points
    Bvwz = spdiagm(Nwz_in, Nvz_t - 1, diagpos => ones(Nwz_in))

    # Map from Npy+2 points to Nvy_t-1 points (vy faces)
    Bkwz = copy(Bmap)


    ## Volumes
    # Volume (area) of pressure control volumes
    Ωp = hzi ⊗ hyi ⊗ hxi

    # Volume (area) of u control volumes
    Ωu = hzi ⊗ hyi ⊗ gxi

    # Volume (area) of v control volumes
    Ωv = hzi ⊗ gyi ⊗ hxi

    # Volume (area) of w control volumes
    Ωw = gzi ⊗ hyi ⊗ hxi

    # Total volumes
    Ω = [Ωu; Ωv; Ωw]
    Ω⁻¹ = 1 ./ Ω

    # Metrics that can be useful for initialization:
    xu = reshape(ones(Nuz_in) ⊗ ones(Nuy_in) ⊗ xin, Nux_in, Nuy_in, Nuz_in)
    yu = reshape(ones(Nuz_in) ⊗ yp ⊗ ones(Nux_in), Nux_in, Nuy_in, Nuz_in)
    zu = reshape(zp ⊗ ones(Nuy_in) ⊗ ones(Nux_in), Nux_in, Nuy_in, Nuz_in)

    xv = reshape(ones(Nvz_in) ⊗ ones(Nvy_in) ⊗ xp, Nvx_in, Nvy_in, Nvz_in)
    yv = reshape(ones(Nvz_in) ⊗ yin ⊗ ones(Nvx_in), Nvx_in, Nvy_in, Nvz_in)
    zv = reshape(zp ⊗ ones(Nvy_in) ⊗ ones(Nvx_in), Nvx_in, Nvy_in, Nvz_in)

    xw = reshape(ones(Nwz_in) ⊗ ones(Nwy_in) ⊗ xp, Nwx_in, Nwy_in, Nwz_in)
    yw = reshape(ones(Nwz_in) ⊗ yp ⊗ ones(Nwx_in), Nwx_in, Nwy_in, Nwz_in)
    zw = reshape(zin ⊗ ones(Nwy_in) ⊗ ones(Nwx_in), Nwx_in, Nwy_in, Nwz_in)

    xpp = reshape(ones(Nz) ⊗ ones(Ny) ⊗ xp, Nx, Ny, Nz)
    ypp = reshape(ones(Nz) ⊗ yp ⊗ ones(Nx), Nx, Ny, Nz)
    zpp = reshape(zp ⊗ ones(Ny) ⊗ ones(Nx), Nx, Ny, Nz)

    # Indices of unknowns in velocity vector
    indu = 1:Nu
    indv = Nu .+ (1:Nv)
    indw = Nu + Nv .+ (1:Nw)
    indV = 1:NV
    indp = NV .+ (1:Np)

    Grid{T,3}(;
        Nx,
        Ny,
        Nz,
        xlims,
        ylims,
        zlims,
        x,
        y,
        z,
        xp,
        yp,
        zp,
        hx,
        hy,
        hz,
        gx,
        gy,
        gz,
        Npx,
        Npy,
        Npz,
        Np,
        Nux_in,
        Nux_b,
        Nux_t,
        Nuy_in,
        Nuy_b,
        Nuy_t,
        Nuz_in,
        Nuz_b,
        Nuz_t,
        Nvx_in,
        Nvx_b,
        Nvx_t,
        Nvy_in,
        Nvy_b,
        Nvy_t,
        Nvz_in,
        Nvz_b,
        Nvz_t,
        Nwx_in,
        Nwx_b,
        Nwx_t,
        Nwy_in,
        Nwy_b,
        Nwy_t,
        Nwz_in,
        Nwz_b,
        Nwz_t,
        Nu,
        Nv,
        Nw,
        NV,
        Ωp,
        Ω,
        Ω⁻¹,
        hxi,
        hyi,
        hzi,
        hxd,
        hyd,
        hzd,
        gxi,
        gyi,
        gzi,
        gxd,
        gyd,
        gzd,
        Buvy,
        Bvux,
        Buwz,
        Bwux,
        Bvwz,
        Bwvy,
        Bkux,
        Bkvy,
        Bkwz,
        xin,
        yin,
        zin,
        xu,
        yu,
        zu,
        xv,
        yv,
        zv,
        xw,
        yw,
        zw,
        xpp,
        ypp,
        zpp,
        indu,
        indv,
        indw,
        indV,
        indp,
    )
end
