# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

using GLMakie
using IncompressibleNavierStokes

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using CUDA;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);

# Viscosity model
Re = T(10_000)

# A 2D grid is a Cartesian product of two vectors
n = 1024
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)

# Build setup and assemble operators
setup = Setup(x...; Re, ArrayType);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
pressure_solver = SpectralPressureSolver(setup);

u₀ = random_field(setup, T(0); pressure_solver);
u = u₀

function filter_plot(state, setup, les, comp; resolution = (1200, 600))
    (; boundary_conditions, grid) = setup
    (; dimension, xlims, x, xp, Ip) = grid
    D = dimension()
    xf = Array.(getindex.(setup.grid.xp, Ip.indices))
    xfbar = Array.(getindex.(les.grid.xp, les.grid.Ip.indices))
    (; u, t) = state[]
    ω = IncompressibleNavierStokes.vorticity(u, setup)
    ωp = IncompressibleNavierStokes.interpolate_ω_p(ω, setup)
    _f = Array(ωp)[Ip]
    f = @lift begin
        sleep(0.001)
        (; u, t) = $state
        IncompressibleNavierStokes.apply_bc_u!(u, t, setup)
        IncompressibleNavierStokes.vorticity!(ω, u, setup)
        IncompressibleNavierStokes.interpolate_ω_p!(ωp, ω, setup)
        copyto!(_f, view(ωp, Ip))
    end
    ubar = zero.(IncompressibleNavierStokes.face_average(u, les, comp))
    ωbar = IncompressibleNavierStokes.vorticity(ubar, les)
    ωpbar = IncompressibleNavierStokes.interpolate_ω_p(ωbar, les)
    _g = Array(ωpbar)[les.grid.Ip]
    g = @lift begin
        (; u, t) = $state
        IncompressibleNavierStokes.face_average!(ubar, u, les, comp)
        IncompressibleNavierStokes.apply_bc_u!(ubar, t, les)
        IncompressibleNavierStokes.vorticity!(ωbar, ubar, les)
        IncompressibleNavierStokes.interpolate_ω_p!(ωpbar, ωbar, les)
        copyto!(_g, view(ωpbar, les.grid.Ip))
    end
    lims = @lift IncompressibleNavierStokes.get_lims($f)
    fig = Figure(; resolution)
    ax, hm = heatmap(
        fig[1, 1],
        xf...,
        f;
        colorrange = lims,
        axis = (;
            title = "$(setup.grid.N .- 2)",
            aspect = DataAspect(),
            xlabel = "x",
            ylabel = "y",
            limits = (xlims[1]..., xlims[2]...),
        ),
    )
    ax, hm = heatmap(
        fig[1, 2],
        xfbar...,
        g;
        colorrange = lims,
        axis = (;
            title = "$(les.grid.N .- 2)",
            aspect = DataAspect(),
            xlabel = "x",
            # ylabel = "y",
            # yticksvisible = false,
            # yticklabelsvisible = false,
            limits = (xlims[1]..., xlims[2]...),
        ),
    )
    # Colorbar(fig[1, 3], hm)
    display(fig)
    fig
end

set_theme!(
    theme_black();
    # colormap = :plasma,
    # colormap = :lajolla,
    # colormap = :inferno,
    # colormap = :hot,
    # colormap = :magma,
    # colormap = :Oranges,
    # colormap = :Reds,
    # colormap = :YlOrRd_9,
    colormap = :seaborn_icefire_gradient,
)

comp = 8
les = Setup(x[1][1:comp:end], x[2][1:comp:end]; Re, ArrayType)

# Solve unsteady problem
(; u)outputs = solve_unsteady(
    setup,
    u₀,
    (T(0), T(2e0));
    Δt = T(1e-4),
    pressure_solver,
    inplace = true,
    processors = (
        # realtimeplotter(; setup, nupdate = 10, docolorbar = false, displayupdates = false),
        # animator(
        #     setup,
        #     "filtered.mp4";
        #     plotter = processor(state -> filter_plot(state, setup, les, comp; resolution = (1200, 600))),
        #     nupdate = 100,
        # ),
        processor(
            state -> filter_plot(state, setup, les, comp; resolution = (1200, 600));
            nupdate = 10,
        ),
        # energy_history_plotter(setup; nupdate = 20, displayfig = false),
        # energy_spectrum_plotter(setup; nupdate = 10, displayfig = false),
        ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
        ## fieldsaver(setup; nupdate = 10),
        timelogger(; nupdate = 10),
    ),
);
