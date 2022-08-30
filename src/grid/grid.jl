"""
    Grid{T,N}()

Nonuniform Cartesian grid of dimension `N` and floating point type `T`.
"""
Base.@kwdef struct Grid{T,N}
    order4::Bool = false                     # Use 4th order in space (otherwise 2nd order)
    α::T = 81                                # Richardson extrapolation factor = 3^4
    β::T = 9 // 8                            # Interpolation factor

    Nx::Int = 10                             # Number of x-volumes
    Ny::Int = 10                             # Number of y-volumes
    Nz::Int = 1                              # Number of z-volumes (if any)
    xlims::Tuple{T,T} = (0, 1)               # Horizontal limits (left, right)
    ylims::Tuple{T,T} = (0, 1)               # Vertical limits (bottom, top)
    zlims::Tuple{T,T} = (0, 1)               # Depth limits (back, front)

    x::Vector{T} = T[]                       # Vector of x-points
    y::Vector{T} = T[]                       # Vector of y-points
    z::Vector{T} = T[]                       # Vector of z-points
    xp::Vector{T} = T[]
    yp::Vector{T} = T[]
    zp::Vector{T} = T[]

    # Number of pressure points in each dimension
    Npx::Int = 0
    Npy::Int = 0
    Npz::Int = 0

    Nux_in::Int = 0
    Nux_b::Int = 0
    Nux_t::Int = 0
    Nuy_in::Int = 0
    Nuy_b::Int = 0
    Nuy_t::Int = 0
    Nuz_in::Int = 0
    Nuz_b::Int = 0
    Nuz_t::Int = 0

    Nvx_in::Int = 0
    Nvx_b::Int = 0
    Nvx_t::Int = 0
    Nvy_in::Int = 0
    Nvy_b::Int = 0
    Nvy_t::Int = 0
    Nvz_in::Int = 0
    Nvz_b::Int = 0
    Nvz_t::Int = 0

    Nwx_in::Int = 0
    Nwx_b::Int = 0
    Nwx_t::Int = 0
    Nwy_in::Int = 0
    Nwy_b::Int = 0
    Nwy_t::Int = 0
    Nwz_in::Int = 0
    Nwz_b::Int = 0
    Nwz_t::Int = 0

    # Number of points in solution vector
    Nu::Int = 0
    Nv::Int = 0
    Nw::Int = 0
    NV::Int = 0
    Np::Int = 0

    Ωp::Vector{T} = T[]
    Ω::Vector{T} = T[]
    Ω⁻¹::Vector{T} = T[]

    # For order4
    Ωux::Vector{T} = T[]
    Ωvx::Vector{T} = T[]
    Ωwx::Vector{T} = T[]
    Ωuy::Vector{T} = T[]
    Ωvy::Vector{T} = T[]
    Ωwy::Vector{T} = T[]
    Ωuz::Vector{T} = T[]
    Ωvz::Vector{T} = T[]
    Ωwz::Vector{T} = T[]

    hx::Vector{T} = T[]
    hy::Vector{T} = T[]
    hz::Vector{T} = T[]
    hxi::Vector{T} = T[]
    hyi::Vector{T} = T[]
    hzi::Vector{T} = T[]
    hxd::Vector{T} = T[]
    hyd::Vector{T} = T[]
    hzd::Vector{T} = T[]
    gx::Vector{T} = T[]
    gy::Vector{T} = T[]
    gz::Vector{T} = T[]
    gxi::Vector{T} = T[]
    gyi::Vector{T} = T[]
    gzi::Vector{T} = T[]
    gxd::Vector{T} = T[]
    gyd::Vector{T} = T[]
    gzd::Vector{T} = T[]

    Buvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Buwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bwux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bwvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Bkux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bkvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bkwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    xin::Vector{T} = T[]
    yin::Vector{T} = T[]
    zin::Vector{T} = T[]

    # Separate grids for u, v, and p
    xu::Array{T,N} = zeros(T, fill(0, N)...)
    xv::Array{T,N} = zeros(T, fill(0, N)...)
    xw::Array{T,N} = zeros(T, fill(0, N)...)
    yu::Array{T,N} = zeros(T, fill(0, N)...)
    yv::Array{T,N} = zeros(T, fill(0, N)...)
    yw::Array{T,N} = zeros(T, fill(0, N)...)
    zu::Array{T,N} = zeros(T, fill(0, N)...)
    zv::Array{T,N} = zeros(T, fill(0, N)...)
    zw::Array{T,N} = zeros(T, fill(0, N)...)
    xpp::Array{T,N} = zeros(T, fill(0, N)...)
    ypp::Array{T,N} = zeros(T, fill(0, N)...)
    zpp::Array{T,N} = zeros(T, fill(0, N)...)

    # Ranges
    indu::UnitRange{Int} = 0:0
    indv::UnitRange{Int} = 0:0
    indw::UnitRange{Int} = 0:0
    indV::UnitRange{Int} = 0:0
    indp::UnitRange{Int} = 0:0

    hx3::Vector{T} = T[]
    hy3::Vector{T} = T[]
    hxi3::Vector{T} = T[]
    hyi3::Vector{T} = T[]
    gxi3::Vector{T} = T[]
    gyi3::Vector{T} = T[]

    hxd13::Vector{T} = T[]
    hxd3::Vector{T} = T[]
    hyd13::Vector{T} = T[]
    hyd3::Vector{T} = T[]
    gxd13::Vector{T} = T[]
    gxd3::Vector{T} = T[]
    gyd13::Vector{T} = T[]
    gyd3::Vector{T} = T[]

    Ωux1::Vector{T} = T[]
    Ωux3::Vector{T} = T[]
    Ωuy1::Vector{T} = T[]
    Ωuy3::Vector{T} = T[]
    Ωvx1::Vector{T} = T[]
    Ωvx3::Vector{T} = T[]
    Ωvy1::Vector{T} = T[]
    Ωvy3::Vector{T} = T[]
end

"""
    Grid(x, y; boundary_conditions, order4 = false, T = eltype(x))

Create nonuniform Cartesian box mesh `x` × `y` with boundary conditions `boundary_conditions`.
If `order4` is `true`, a fourth order mesh is created.
"""
function Grid(x, y; boundary_conditions, order4 = false, T = eltype(x))
    α = 81

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
    Nux_in -= boundary_conditions.u.x[1] ∈ [:dirichlet, :symmetric]
    Nux_in -= boundary_conditions.u.x[2] ∈ [:dirichlet, :symmetric]
    Nux_in -= boundary_conditions.u.x == (:periodic, :periodic)
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
    Nvy_in = Ny + 1         # Inner points
    Nvy_in -= boundary_conditions.v.y[1] ∈ [:dirichlet, :symmetric]
    Nvy_in -= boundary_conditions.v.y[2] ∈ [:dirichlet, :symmetric]
    Nvy_in -= boundary_conditions.v.y == (:periodic, :periodic)
    Nvy_t = Nvy_in + Nvy_b # Total number

    # Total number
    Nv = Nvx_in * Nvy_in

    # Total number of velocity points
    NV = Nu + Nv

    ## For a grid with three times larger volumes:
    if order4
        hx3 = zeros(Nx)
        hx3[2:(end-1)] = hx[1:(end-2)] + hx[2:(end-1)] + hx[3:end]
        if boundary_conditions.u.x[1] == :periodic &&
           boundary_conditions.u.x[2] == :periodic
            hx3[1] = hx[end] + hx[1] + hx[2]
            hx3[end] = hx[end-1] + hx[end] + hx[1]
        else
            hx3[1] = 2 * hx[1] + hx[2]
            hx3[end] = hx[end-1] + 2 * hx[end]
        end

        hy3 = zeros(Ny)
        hy3[2:(end-1)] = hy[1:(end-2)] + hy[2:(end-1)] + hy[3:end]
        if boundary_conditions.v.y[1] == :periodic &&
           boundary_conditions.v.y[2] == :periodic
            hy3[1] = hy[end] + hy[1] + hy[2]
            hy3[end] = hy[end-1] + hy[end] + hy[1]
        else
            hy3[1] = 2 * hy[1] + hy[2]
            hy3[end] = hy[end-1] + 2 * hy[end]
        end

        hxi3 = copy(hx3)
        hyi3 = copy(hy3)

        # Distance between pressure points
        gx3 = zeros(Nx + 1)
        gx3[3:(Nx-1)] = gx[2:(end-3)] + gx[3:(end-2)] + gx[4:(end-1)]
        if boundary_conditions.u.x[1] == :periodic &&
           boundary_conditions.u.x[2] == :periodic
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
        gy3 = zeros(Ny + 1)
        gy3[3:(Ny-1)] = gy[2:(end-3)] + gy[3:(end-2)] + gy[4:(end-1)]
        if boundary_conditions.v.y[1] == :periodic &&
           boundary_conditions.v.y[2] == :periodic
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

    # gxd: differentiation
    gxd = copy(gx)
    gxd[1] = hx[1]
    gxd[end] = hx[end]

    # hxi: integration and hxd: differentiation
    # Map to find suitable size
    hxi = copy(hx)

    # Restrict Nx+2 to Nux_in+1 points
    if boundary_conditions.u.x == (:dirichlet, :dirichlet)
        xin = x[2:(end-1)]
        hxd = copy(hx)
        gxi = gx[2:(end-1)]
        diagpos = 1

        if order4
            hxd3 = [hx3[1]; hx3; hx3[end]]
            hxd13 = [hx[1]; hx; hx[end]]
            gxd3 = [2 * gx[1] + gx[2] + gx[3]; gx3; 2 * gx[end] + gx[end-1] + gx[end-2]]
            gxd13 = [gx[2]; 2 * gx[1]; gx[2:(end-1)]; 2 * gx[end]; gx[end-1]]
            gxi3 = gx3[2:(end-1)]
        end
    elseif boundary_conditions.u.x == (:dirichlet, :pressure)
        xin = x[2:end]
        hxd = [hx; hx[end]]
        gxi = gx[2:end]
        diagpos = 1
    elseif boundary_conditions.u.x == (:pressure, :dirichlet)
        xin = x[1:(end-1)]
        hxd = [hx[1]; hx]
        gxi = gx[1:(end-1)]
        diagpos = 0
    elseif boundary_conditions.u.x == (:pressure, :pressure)
        xin = x[1:end]
        hxd = [hx[1]; hx; hx[end]]
        gxi = copy(gx)
        diagpos = 0
    elseif boundary_conditions.u.x == (:periodic, :periodic)
        xin = x[1:(end-1)]
        hxd = [hx[end]; hx]
        gxi = [gx[1] + gx[end]; gx[2:(end-1)]]
        gxd[1] = (hx[1] + hx[end]) / 2
        gxd[end] = (hx[1] + hx[end]) / 2
        diagpos = 0

        if order4
            hxd3 = [hx3[end-1]; hx3[end]; hx3; hx3[1]]
            hxd13 = [hx[end-1]; hx[end]; hx; hx[1]]
            gxd3 = [gx3[end-1]; gx3; gx3[2]]
            gxd13 = [gx[end-1]; gx[1] + gx[end]; gx[2:(end-1)]; gx[end] + gx[1]; gx[2]]
            gxi3 = gx3[1:(end-1)]
        end
    end

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
    if boundary_conditions.v.y == (:dirichlet, :dirichlet)
        yin = y[2:(end-1)]
        hyd = copy(hy)
        gyi = gy[2:(end-1)]
        diagpos = 1

        if order4
            hyd3 = [hy3[1]; hy3; hy3[end]]
            hyd13 = [hy[1]; hy; hy[end]]
            gyd3 = [2 * gy[1] + gy[2] + gy[3]; gy3; 2 * gy[end] + gy[end-1] + gy[end-2]]
            gyd13 = [gy[2]; 2 * gy[1]; gy[2:(end-1)]; 2 * gy[end]; gy[end-1]]
            gyi3 = gy3[2:(end-1)]
        end
    elseif boundary_conditions.v.y == (:dirichlet, :pressure)
        yin = y[2:end]
        hyd = [hy; hy[end]]
        gyi = gy[2:end]
        diagpos = 1
    elseif boundary_conditions.v.y == (:pressure, :dirichlet)
        yin = y[1:(end-1)]
        hyd = [hy[1]; hy]
        gyi = gy[1:(end-1)]
        diagpos = 0
    elseif boundary_conditions.v.y == (:pressure, :pressure)
        yin = y[1:end]
        hyd = [hy[1]; hy; hy[end]]
        gyi = copy(gy)
        diagpos = 0
    elseif boundary_conditions.v.y == (:periodic, :periodic)
        yin = y[1:(end-1)]
        hyd = [hy[end]; hy]
        gyi = [gy[1] + gy[end]; gy[2:(end-1)]]
        gyd[1] = (hy[1] + hy[end]) / 2
        gyd[end] = (hy[1] + hy[end]) / 2
        diagpos = 0

        if order4
            hyd3 = [hy3[end-1]; hy3[end]; hy3; hy3[1]]
            hyd13 = [hy[end-1]; hy[end]; hy; hy[1]]
            gyd3 = [gy3[end-1]; gy3; gy3[2]]
            gyd13 = [gy[end-1]; gy[1] + gy[end]; gy[2:(end-1)]; gy[end] + gy[1]; gy[2]]
            gyi3 = gy3[1:(end-1)]
        end
    end

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

    if order4
        # Differencing for second order operators on the fourth order mesh
        Ωux1 = hyi ⊗ hxd13
        Ωuy1 = gyd13 ⊗ gxi
        Ωvx1 = gyi ⊗ gxd13
        Ωvy1 = hyd13 ⊗ hxi

        # Volume (area) of pressure control volumes
        Ωp3 = hyi3 ⊗ hxi3

        # Volume (area) of u-vel control volumes
        Ωu3 = hyi3 ⊗ gxi3

        # Volume (area) of v-vel control volumes
        Ωv3 = gyi3 ⊗ hxi3

        # Volume (area) of dudx control volumes
        Ωux3 = hyi3 ⊗ hxd3

        # Volume (area) of dudy control volumes
        Ωuy3 = gyd3 ⊗ gxi3

        # Volume (area) of dvdx control volumes
        Ωvx3 = gyi3 ⊗ gxd3

        # Volume (area) of dvdy control volumes
        Ωvy3 = hyd3 ⊗ hxi3

        Ωu1 = copy(Ωu)
        Ωv1 = copy(Ωv)

        Ωu = α * Ωu1 - Ωu3
        Ωv = α * Ωv1 - Ωv3
        Ω = [Ωu; Ωv]
        Ω⁻¹ = 1 ./ Ω

        Ωux = @. α * Ωux1 - Ωux3
        Ωuy = @. α * Ωuy1 - Ωuy3
        Ωvx = @. α * Ωvx1 - Ωvx3
        Ωvy = @. α * Ωvy1 - Ωvy3
    end

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

    ## Store quantities in the structure
    params = (;
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

    if order4
        params = (;
            params...,
            hx3,
            hy3,
            hxi3,
            hyi3,
            gxi3,
            gyi3,
            hxd13,
            hxd3,
            hyd13,
            hyd3,
            gxd13,
            gxd3,
            gyd13,
            gyd3,
            Ωux1,
            Ωux3,
            Ωuy1,
            Ωuy3,
            Ωvx1,
            Ωvx3,
            Ωvy1,
            Ωvy3,
        )
    end

    Grid{T,2}(; α, order4, Nx, Ny, xlims, ylims, x, y, xp, yp, hx, hy, gx, gy, params...)
end

"""
    Grid(x, y, z; boundary_conditions, order4 = false, T = eltype(x))

Create nonuniform Cartesian box mesh `x` × `y` × `z` with boundary conditions `boundary_conditions`.
If `order4` is `true`, a fourth order mesh is created.
"""
function Grid(x, y, z; boundary_conditions, order4 = false, T = eltype(x))
    order4 && error("Fourth order grids not yet implemented for 3D")
    α = 81

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
    Nux_in -= boundary_conditions.u.x[1] ∈ [:dirichlet, :symmetric]
    Nux_in -= boundary_conditions.u.x[2] ∈ [:dirichlet, :symmetric]
    Nux_in -= boundary_conditions.u.x == (:periodic, :periodic)
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
    Nvy_in = Ny + 1         # Inner points
    Nvy_in -= boundary_conditions.v.y[1] ∈ [:dirichlet, :symmetric]
    Nvy_in -= boundary_conditions.v.y[2] ∈ [:dirichlet, :symmetric]
    Nvy_in -= boundary_conditions.v.y == (:periodic, :periodic)
    Nvy_t = Nvy_in + Nvy_b # Total number

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
    Nwz_in = Nz + 1         # Inner points
    Nwz_in -= boundary_conditions.w.z[1] ∈ [:dirichlet, :symmetric]
    Nwz_in -= boundary_conditions.w.z[2] ∈ [:dirichlet, :symmetric]
    Nwz_in -= boundary_conditions.w.z == (:periodic, :periodic)
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
    boundary_conditions.u.x == (:dirichlet, :dirichlet) && (diagpos = 1)
    boundary_conditions.u.x == (:dirichlet, :pressure) && (diagpos = 1)
    boundary_conditions.u.x == (:pressure, :dirichlet) && (diagpos = 0)
    boundary_conditions.u.x == (:pressure, :pressure) && (diagpos = 0)
    boundary_conditions.u.x == (:periodic, :periodic) && (diagpos = 0)

    Bmap = spdiagm(Nux_in + 1, Nx + 2, diagpos => ones(Nux_in + 1))
    Bmap_x_xin = spdiagm(Nux_in, Nx + 1, diagpos => ones(Nux_in))
    hxd = Bmap * hxd
    gxi = Bmap_x_xin * gx
    xin = Bmap_x_xin * x

    if boundary_conditions.u.x == (:periodic, :periodic)
        gxd[1] = (hx[1] + hx[end]) / 2
        gxd[end] = gxd[1]
        gxi[1] = gxd[1]
        hxd[1] = hx[end]
    end

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
    boundary_conditions.v.y == (:dirichlet, :dirichlet) && (diagpos = 1)
    boundary_conditions.v.y == (:dirichlet, :pressure) && (diagpos = 1)
    boundary_conditions.v.y == (:pressure, :dirichlet) && (diagpos = 0)
    boundary_conditions.v.y == (:pressure, :pressure) && (diagpos = 0)
    boundary_conditions.v.y == (:periodic, :periodic) && (diagpos = 0)

    Bmap = spdiagm(Nvy_in + 1, Ny + 2, diagpos => ones(Nvy_in + 1))
    Bmap_y_yin = spdiagm(Nvy_in, Ny + 1, diagpos => ones(Nvy_in))

    hyd = Bmap * hyd
    gyi = Bmap_y_yin * gy
    yin = Bmap_y_yin * y

    if boundary_conditions.v.y == (:periodic, :periodic)
        gyd[1] = (hy[1] + hy[end]) / 2
        gyd[end] = gyd[1]
        gyi[1] = gyd[1]
        hyd[1] = hy[end]
    end

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
    boundary_conditions.w.z == (:dirichlet, :dirichlet) && (diagpos = 1)
    boundary_conditions.w.z == (:dirichlet, :pressure) && (diagpos = 1)
    boundary_conditions.w.z == (:pressure, :dirichlet) && (diagpos = 0)
    boundary_conditions.w.z == (:pressure, :pressure) && (diagpos = 0)
    boundary_conditions.w.z == (:periodic, :periodic) && (diagpos = 0)

    shape = (Nwz_in + 1, Nz + 2)
    Bmap = spdiagm(shape..., diagpos => ones(Nwz_in + 1))
    Bmap_z_zin = spdiagm(Nwz_in, Nz + 1, diagpos => ones(Nwz_in))
    hzd = Bmap * hzd
    gzi = Bmap_z_zin * gz
    zin = Bmap_z_zin * z

    if boundary_conditions.w.z == (:periodic, :periodic)
        gzd[1] = (hz[1] + hz[end]) / 2
        gzd[end] = gzd[1]
        gzi[1] = gzd[1]
        hzd[1] = hz[end]
    end

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
        order4,
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
