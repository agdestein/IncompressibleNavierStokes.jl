# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Backward Facing Step - 2D
#
# In this example we consider a channel with walls at the top and bottom, and a
# step at the left with a parabolic inflow. Initially the velocity is an
# extension of the inflow, but as time passes the velocity finds a new steady
# state.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "BackwardFacingStep2D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(3_000)

# Boundary conditions: steady inflow on the top half
U(dim, x, y, t) =
    dim() == 1 && y ≥ 0 ? 24y * (one(x) / 2 - y) : zero(x) + randn(typeof(x)) / 1_000
dUdt(dim, x, y, t) = zero(x)
boundary_conditions = (
    ## x left, x right
    (DirichletBC(U, dUdt), PressureBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC()),
)

# A 2D grid is a Cartesian product of two vectors. Here we refine the grid near
# the walls.
x = LinRange(T(0), T(10), 301)
y = cosine_grid(-T(0.5), T(0.5), 51)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; Re, boundary_conditions, ArrayType);

pressure_solver = CGPressureSolverManual(setup);

# Initial conditions (extend inflow)
u₀, p₀ =
    create_initial_conditions(setup, (dim, x, y) -> U(dim, x, y, zero(x)); pressure_solver);
u, p = copy.(u₀), copy(p₀)

# Solve steady state problem
## u, p = solve_steady_state(setup, u₀, p₀);

# Solve unsteady problem
u, p, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    # u, p,
    (T(0), T(7));
    Δt = T(0.002),
    pressure_solver,
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            ## plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            nupdate = 1,
        ),
        ## anim = animator(; setup, path = "vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = "output/$name", filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(setup, u, p, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, u)

# Plot vorticity
plot_vorticity(setup, u)

# Plot streamfunction
plot_streamfunction(setup, u)
