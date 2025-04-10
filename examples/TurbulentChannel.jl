# ## Turbulent channel flow

if false
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CairoMakie
using CUDA
using CUDSS

function sectionplot(state; setup, component)
    state isa Observable || (state = Observable(state))
    (; xu) = setup.grid
    xplot = xu[component][1][2:(end-1)] |> Array
    yplot = xu[component][2][2:(end-1)] |> Array
    zplot = xu[component][3] |> Array
    imid = div(setup.grid.N[1], 2)
    jmid = div(setup.grid.N[2], 2)
    u_xz = map(state) do (; u)
        # View u at y = 0.5
        u[2:(end-1), jmid, :, component] |> Array
    end
    u_yz = map(state) do (; u)
        # View u at given x
        u[imid, 2:(end-1), :, component] |> Array
    end
    fig = Figure(; size = (800, 300))
    ax_yz = Axis(fig[1, 1]; title = "u$(component) at x = 2.5", xlabel = "y", ylabel = "z")
    ax_xz = Axis(
        fig[1, 2];
        title = "u$(component) at y = 0.5",
        xlabel = "x",
        yticklabelsvisible = false,
        yticksvisible = false,
    )
    heatmap!(ax_xz, xplot, zplot, u_xz)
    heatmap!(ax_yz, yplot, zplot, u_yz)
    linkyaxes!(ax_xz, ax_yz)
    colsize!(fig.layout, 1, Relative(1 / 4))
    fig
end

f = one(Float32)
n = 50
setup = Setup(;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
    ),
    x = (range(0f, 5f, 5n + 1), range(0f, 1f, n + 1), tanh_grid(0f, 1f, n + 1)),
    Re = 6000f,
    bodyforce = (dim, x, y, z, t) ->
        (dim == 1) * 10 * 4 * z * (1 - z) + (dim == 2) * sinpi(10x) / 5,
    issteadybodyforce = true,
    backend = CUDABackend(),
)
psolver = default_psolver(setup)
ustart = velocityfield(
    setup,
    (dim, x, y, z) ->
        (dim == 1) * 4 * z * (1 - z) +
        (dim == 2) * sinpi(10x) * sinpi(5z) / 10 +
        (dim == 3) * randn(typeof(x)),
);

heatmap(setup.bodyforce[:, :, 10, 1] |> Array)
heatmap(setup.bodyforce[:, 10, :, 2] |> Array)
plotgrid(setup.grid.x[2] |> Array, setup.grid.x[3] |> Array)

sol, outputs = solve_unsteady(;
    setup,
    psolver,
    ustart,
    tlims = (0f, 1f),
    processors = (;
        logger = timelogger(; nupdate = 1),
        plotter = realtimeplotter(;
            plot = sectionplot,
            setup,
            nupdate = 1,
            displayupdates = true,
            component = 1,
        ),
    ),
);

sectionplot(sol; setup, component = 1)
