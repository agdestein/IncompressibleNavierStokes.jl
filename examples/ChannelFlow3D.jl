# # Turbulent channel flow

if false
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CairoMakie
using WGLMakie
using WriteVTK
## using CUDA, CUDSS;

function sectionplot(state; setup, component, qrange = (0, 500))
    state isa Observable || (state = Observable(state))
    (; xu) = setup
    xplot = xu[component][1][2:(end-1)] |> Array
    yplot = xu[component][2] |> Array
    zplot = xu[component][3][2:(end-1)] |> Array
    imid = div(setup.N[1], 2)
    kmid = div(setup.N[3], 2)
    q = scalarfield(setup)
    u_xy = map(state) do (; u)
        ## View u at y = 0.5
        ## u[2:end-1, :, kmid, component] |> Array
        ## unorm = @. sqrt(
        ##     u[2:(end-1), :, kmid, 1]^2 +
        ##     u[2:(end-1), :, kmid, 2]^2 +
        ##     u[2:(end-1), :, kmid, 3]^2,
        ## )
        IncompressibleNavierStokes.qcrit!(q, u, setup)
        unorm = q[2:(end-1), :, kmid]
        unorm |> Array
    end
    u_zy = map(state) do (; u)
        ## View u at given x
        ## u[imid, :, 2:end-1, component] |> Array
        ## unorm = @. sqrt(
        ##     u[imid, :, 2:(end-1), 1]^2 +
        ##     u[imid, :, 2:(end-1), 2]^2 +
        ##     u[imid, :, 2:(end-1), 3]^2,
        ## )
        IncompressibleNavierStokes.qcrit!(q, u, setup)
        unorm = q[imid, :, 2:(end-1)]
        unorm = unorm'
        unorm |> Array
    end
    fig = Figure(; size = (800, 300))
    ax_zy = Axis(fig[1, 1]; title = "u$(component) at x = 2pi", xlabel = "z", ylabel = "y")
    ax_xy = Axis(
        fig[1, 2];
        title = "u$(component) at z = 1",
        xlabel = "x",
        ylabel = "y",
        ## yticklabelsvisible = false,
        ## yticksvisible = false,
    )
    colorrange = qrange
    heatmap!(ax_xy, xplot, yplot, u_xy; colorrange)
    hm = heatmap!(ax_zy, zplot, yplot, u_zy; colorrange)
    Colorbar(fig[1, 3], hm)
    ## linkyaxes!(ax_xy, ax_yz)
    colsize!(fig.layout, 1, Relative(1 / 4))
    fig
end

# Precision
T = Float64

# Domain
xlims = T(0), T(4 * π)
ylims = T(0), T(2)
zlims = T(0), T(4 / 3 * π)

# Grid
## nx, ny, nz = 32, 16, 16
nx, ny, nz = 48, 24, 24
## nx, ny, nz = 64, 32, 32
## nx, ny, nz = 128, 64, 64
## nx, ny, nz = 200, 100, 100
## nx, ny, nz = 256, 128, 128
## nx, ny, nz = 512, 512, 256

setup = Setup(;
    boundary_conditions = (;
        u = (
            (PeriodicBC(), PeriodicBC()),
            (DirichletBC(), DirichletBC()),
            (PeriodicBC(), PeriodicBC()),
        )
    ),
    ## x = (range(xlims..., nx + 1), tanh_grid(ylims..., ny + 1), range(zlims..., nz + 1)),
    x = (range(xlims..., nx + 1), range(ylims..., ny + 1), range(zlims..., nz + 1)),
    ## backend = CUDABackend(),
);

# AMGX solver (for NVidia GPUs)
## AMGX_stuff = amgx_setup();
## psolver = psolver_cg_AMGX(setup; stuff = AMGX_stuff);

# Direct pressure solver
## @time psolver = default_psolver(setup);

# Discrete transform solver (FFT/DCT)
psolver = psolver_transform(setup);

## using Random
## let
##     u = randn!(vectorfield(setup))
##     IncompressibleNavierStokes.apply_bc_u!(u, 0.0, setup)
##     up = project(u, setup; psolver)
##     IncompressibleNavierStokes.apply_bc_u!(up, 0.0, setup)
##     divergence(u, setup) |> extrema
##     divergence(up, setup) |> extrema
##     # divergence(up, setup) |> x -> sum(abs, x) / length(x)
## end

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(f, state, t; setup, cache, viscosity)
    navierstokes!(f, state, t; setup, cache, viscosity)
    @. f.u[:, :, :, 1] += 1 # Force is 1 in direction x
    ## wale_closure!(f.u, u, params, cache.wale, setup)
end

# Tell IncompressibleNavierStokes how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
function IncompressibleNavierStokes.get_cache(::typeof(force!), setup)
    ## f(dim, x, y, z) = (dim == 1) * one(x)
    ## bodyforce = velocityfield(setup, f; psolver, doproject = false)
    ## (; bodyforce)
    ## wale = get_cache(wale_closure, setup)
    ## (; bodyforce, wale)
    nothing
end

# We also need to tell how to propos the time step sizes for our given force.
# We just fall back to the default one.
IncompressibleNavierStokes.propose_timestep(::typeof(force!), state, setup, params) =
    IncompressibleNavierStokes.propose_timestep(navierstokes!, state, setup, params)

Re_tau = 180 |> T
Re_m = 2800 |> T
Re_ratio = Re_m / Re_tau

u = let
    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]
    Lz = zlims[2] - zlims[1]
    C = T(9 / 8) * Re_ratio
    E = T(1 / 10) * Re_ratio ## 10% of average mean velocity
    function U(dim, x, y, z)
        ux =
            C * (1 - (y - Ly / 2)^8) +
            E * Lx / 2 * sinpi(y) * cospi(4 * x / Lx) * sinpi(2 * z / Lz)
        uy = -E * (1 - cospi(y)) * sinpi(4 * x / Lx) * sinpi(2 * z / Lz)
        uz = -E * Lz / 2 * sinpi(4 * x / Lx) * sinpi(y) * cospi(2 * z / Lz)
        (dim == 1) * ux + (dim == 2) * uy + (dim == 3) * uz
    end
    velocityfield(setup, U; psolver)
end;

plotgrid(setup.x[1] |> Array, setup.x[2] |> Array)
plotgrid(setup.x[1] |> Array, setup.x[3] |> Array)
plotgrid(setup.x[2] |> Array, setup.x[3] |> Array)

viscosity = T(1 / 180)
sol, outputs = solve_unsteady(;
    setup,
    force!,
    psolver,
    start = (; u),
    tlims = (0 |> T, 1 |> T),
    params = (;
        viscosity,
        ## wale = T(0.6), # WALE constant
    ),
    processors = (;
        logger = timelogger(; nupdate = 1),
        ## plotter = realtimeplotter(; plot = sectionplot, setup, component = 1, nupdate = 5),
        ## writer = vtk_writer(;
        ##     setup,
        ##     dir = joinpath(@__DIR__, "output", "TCF_INS3"),
        ##     fieldnames = (:qcrit, :velocitynorm),
        ##     # fieldnames = (:qcrit,),
        ##     nupdate = 15,
        ## ),
    ),
);

xp1 = setup.xp[1][2:(end-1)] |> Array
xp2 = setup.xp[2][2:(end-1)] |> Array
xp3 = setup.xp[3][2:(end-1)] |> Array
vtk_grid("output/channel_visc=$(viscosity)", xp1, xp2, xp3) do vtk
    q = qcrit(sol.u, setup);
    # uin = sol.u[2:(end-1), 2:(end-1), 2:(end-1), :]
    unorm = kinetic_energy(sol.u, setup)
    @. unorm = sqrt(2 * unorm)
    # vtk["u"] = (eachslice(uin; dims = 4)...,) .|> Array
    vtk["u"] = view(unorm, setup.Ip) |> Array
    vtk["q"] = view(q, setup.Ip) |> Array
end

# The AMGX solver needs to be closed after use.
## close_amgx(AMGX_stuff)
