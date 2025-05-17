# # Turbulent channel flow

if false
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CairoMakie
using WGLMakie
using WriteVTK

backend = IncompressibleNavierStokes.CPU()
## using CUDA, CUDSS; backend = CUDABackend()

function sectionplot(state; setup, component, qrange = (0, 500))
    state isa Observable || (state = Observable(state))
    (; xu) = setup.grid
    xplot = xu[component][1][2:(end-1)] |> Array
    yplot = xu[component][2] |> Array
    zplot = xu[component][3][2:(end-1)] |> Array
    imid = div(setup.grid.N[1], 2)
    kmid = div(setup.grid.N[3], 2)
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
    # linkyaxes!(ax_xy, ax_yz)
    colsize!(fig.layout, 1, Relative(1 / 4))
    fig
end

# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

# Grid
## nx, ny, nz = 32, 16, 16
nx, ny, nz = 48, 24, 24
## nx, ny, nz = 64, 32, 32
## nx, ny, nz = 128, 64, 64
## nx, ny, nz = 200, 100, 100

setup = Setup(;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    x = (range(xlims..., nx + 1), tanh_grid(ylims..., ny + 1), range(zlims..., nz + 1)),
    Re = 180f,
    backend,
);

# AMGX solver (for NVidia GPUs)
## AMGX_stuff = amgx_setup();
## psolver = psolver_cg_AMGX(setup; stuff = AMGX_stuff);

# Direct pressure solver
@time psolver = default_psolver(setup);

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(f, state, t, params, setup, cache)
    navierstokes!(f, state, t, nothing, setup, nothing)
    f.u .+= cache.bodyforce
    c = wale_closure(u, params, cache.wale, setup)
    f.u .+= c
end

# Tell IncompressibleNavierStokes how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
function IncompressibleNavierStokes.get_cache(::typeof(force!), setup)
    f(dim, x, y, z) = (dim == 1) * one(x)
    bodyforce = velocityfield(setup, f; psolver, doproject = false)
    wale = get_cache(wale_closure, setup)
    (; bodyforce, wale)
end

Re_tau = 180f
Re_m = 2800f
Re_ratio = Re_m / Re_tau

u = let
    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]
    Lz = zlims[2] - zlims[1]
    C = 9f / 8 * Re_ratio
    E = 1f / 10 * Re_ratio # 10% of average mean velocity
    function U(dim, x, y, z)
        ux =
            C * (1 - (y - Ly / 2)^8) +
            E * Lx / 2 * sinpi(y) * cospi(4 * x / Lx) * sinpi(2 * z / Lz)
        uy = -E * (1 - cospi(y)) * sinpi(4 * x / Lx) * sinpi(2 * z / Lz)
        uz = -E * Lz / 2 * sinpi(4 * x / Lx) * sinpi(y) * cospi(2 * z / Lz)
        (dim == 1) * ux + (dim == 2) * uy + (dim == 3) * uz
    end
    velocityfield(setup, U; psolver);
end;

plotgrid(setup.grid.x[1] |> Array, setup.grid.x[2] |> Array)
plotgrid(setup.grid.x[1] |> Array, setup.grid.x[3] |> Array)
plotgrid(setup.grid.x[2] |> Array, setup.grid.x[3] |> Array)

sol, outputs = solve_unsteady(;
    setup,
    force!,
    params = 0.6, # WALE constant
    psolver,
    start = (; u),
    tlims = (0f, 5f),
    processors = (;
        logger = timelogger(; nupdate = 10),
        plotter = realtimeplotter(; plot = sectionplot, setup, component = 1, nupdate = 10),
        ## writer = vtk_writer(;
        ##     setup,
        ##     dir = joinpath(@__DIR__, "output", "TCF_INS3"),
        ##     fieldnames = (:eig2field, :velocity),
        ##     nupdate = 10,
        ## ),
    ),
);

q = qcrit(sol.u, setup)
xp1 = setup.grid.xp[1][2:(end-1)] |> Array
xp2 = setup.grid.xp[2][2:(end-1)] |> Array
xp3 = setup.grid.xp[3][2:(end-1)] |> Array
vtk_grid("uniform", xp1, xp2, xp3) do vtk
    uin = sol.u[2:(end-1), 2:(end-1), 2:(end-1), :]
    vtk["u"] = (eachslice(uin; dims = 4)...,) .|> Array
    vtk["q"] = q[2:(end-1), 2:(end-1), 2:(end-1)] |> Array
end

# The AMGX solver needs to be closed after use.
## close_amgx(AMGX_stuff)
