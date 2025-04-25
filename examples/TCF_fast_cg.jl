## Turbulent channel flow

if false
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CairoMakie
using CUDA
using AMGX
#using WGLMakie

function sectionplot(state; setup, component)
    state isa Observable || (state = Observable(state))
    (; xu) = setup.grid
    xplot = xu[component][1][2:(end-1)] |> Array
    yplot = xu[component][2] |> Array
    zplot = xu[component][3][2:(end-1)] |> Array
    imid = div(setup.grid.N[1], 2)
    kmid = div(setup.grid.N[3], 2)
    u_xy = map(state) do (; u)
        # View u at y = 0.5
        # u[2:end-1, :, kmid, component] |> Array
        unorm = @. sqrt(
            u[2:(end-1), :, kmid, 1]^2 +
            u[2:(end-1), :, kmid, 2]^2 +
            u[2:(end-1), :, kmid, 3]^2,
        )
        unorm |> Array
    end
    u_yz = map(state) do (; u)
        # View u at given x
        # u[imid, :, 2:end-1, component] |> Array

        unorm = @. sqrt(
            u[imid, :, 2:(end-1), 1]^2 +
            u[imid, :, 2:(end-1), 2]^2 +
            u[imid, :, 2:(end-1), 3]^2,
        )
        unorm |> Array
    end
    fig = Figure(; size = (800, 300))
    ax_yz = Axis(fig[1, 1]; title = "u$(component) at x = 2pi", xlabel = "y", ylabel = "z")
    ax_xy = Axis(
        fig[1, 2];
        title = "u$(component) at z = 1",
        xlabel = "x",
        ylabel = "y",
        # yticklabelsvisible = false,
        # yticksvisible = false,
    )
    heatmap!(ax_xy, xplot, yplot, u_xy)
    heatmap!(ax_yz, yplot, zplot, u_yz)
    # linkyaxes!(ax_xy, ax_yz)
    colsize!(fig.layout, 1, Relative(1 / 4))
    fig
end

# Precision
f = one(Float32)
# f = one(Float64)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

# Grid
# nx = 48
# ny = 24
# nz = 24
nx = 128
ny = 64
nz = 64

setup = Setup(;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    x = (range(xlims..., nx + 1), tanh_grid(ylims..., ny + 1), range(zlims..., nz + 1)),
    Re = 180f,
    bodyforce = (dim, x, y, z, t) -> 1 * (dim == 1),
    issteadybodyforce = true,
    backend = CUDABackend(),
);

AMGX_stuff = amgx_setup();
psolver = psolver_cg_AMGX(setup; stuff = AMGX_stuff);

Re_tau = 180f
Re_m = 2800f
Re_ratio = Re_m / Re_tau

ustartfunc = let
    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]
    Lz = zlims[2] - zlims[1]
    C = 9f / 8 * Re_ratio
    E = 1f / 10 * Re_ratio # 10% of average mean velocity
    function icfunc(dim, x, y, z)
        if dim == 1
            C * (1 - (y - Ly / 2)^8) +
            E * Lx / 2 * sinpi(y) * cospi(4 * x / Lx) * sinpi(2 * z / Lz)
        elseif dim == 2
            -E * (1 - cospi(y)) * sinpi(4 * x / Lx) * sinpi(2 * z / Lz)
        elseif dim == 3
            -E * Lz / 2 * sinpi(4 * x / Lx) * sinpi(y) * cospi(2 * z / Lz)
        else
            zero(x)
        end
    end
end

ustart = velocityfield(setup, ustartfunc; psolver);

plotgrid(setup.grid.x[1] |> Array, setup.grid.x[2] |> Array)
plotgrid(setup.grid.x[1] |> Array, setup.grid.x[3] |> Array)
plotgrid(setup.grid.x[2] |> Array, setup.grid.x[3] |> Array)

function volplot(state; setup)
    state isa Observable || (state = Observable(state))
    (; x) = setup.grid
    xplot = x[1][2:(end-1)] |> Array
    yplot = x[2][2:(end-1)] |> Array
    zplot = x[3][2:(end-1)] |> Array
    uplot = observefield(state; setup, fieldname = :velocitynorm)
    # volume(xplot, yplot, zplot, uplot)
    volume(uplot)
end

sol, outputs = solve_unsteady(;
    setup,
    psolver,
    ustart,
    tlims = (0f, 5f),
    processors = (;
        logger = timelogger(; nupdate = 5),
        plotter = realtimeplotter(;
            plot = sectionplot,
            # plot = volplot,
            setup,
            displayupdates = true,
            component = 1,
            nupdate = 2,
            # sleeptime = 0.2,
        ),
        # writer = vtk_writer(;
        #     setup,
        #     dir = joinpath(@__DIR__, "output", "TCF_INS3"),
        #     fieldnames = (:eig2field, :velocity),
        #     nupdate = 10,
        # ),
    ),
);

sectionplot(sol; setup, component = 1)
close_amgx(AMGX_stuff)
