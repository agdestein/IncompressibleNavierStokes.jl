# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Lid-Driven Cavity - 3D
#
# In this example we consider a box with a moving lid. The velocity is initially at rest. The
# solution should reach at steady state equilibrium after a certain time. The same steady
# state should be obtained when solving a steady state problem.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "LidDrivenCavity3D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(1_000)

# A 3D grid is a Cartesian product of three vectors. Here we refine the grid
# near the walls.
x = cosine_grid(T(0), T(1), 25)
y = cosine_grid(T(0), T(1), 25)
z = LinRange(-T(0.2), T(0.2), 11)
plot_grid(x, y, z)

# Boundary conditions: horizontal movement of the top lid
lidvel = (
    (x, y, z, t) -> one(x),
    (x, y, z, t) -> zero(x),
    (x, y, z, t) -> one(x) / 5,
)
dlidveldt = (
    (x, y, z, t) -> zero(x),
    (x, y, z, t) -> zero(x),
    (x, y, z, t) -> zero(x),
)
boundary_conditions = (
    ## x left, x right
    (DirichletBC(), DirichletBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC(lidvel, dlidveldt)),

    ## z bottom, z top
    (PeriodicBC(), PeriodicBC()),
)

# Build setup and assemble operators
setup = Setup(x, y, z; Re, boundary_conditions, ArrayType);

# Time interval
t_start, t_end = tlims = T(0), T(0.2)

# Initial conditions
initial_velocity = (
    (x, y, z) -> zero(x),
    (x, y, z) -> zero(x),
    (x, y, z) -> zero(x),
)
u₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity,
    t_start;
    pressure_solver,
)

# Solve steady state problem
## u, p = solve_steady_state(setup, u₀, p₀; npicard = 5, maxiter = 15);

# Iteration processors
processors = (
    field_plotter(setup; nupdate = 1),
    ## energy_history_plotter(setup; nupdate = 1),
    ## energy_spectrum_plotter(setup; nupdate = 100),
    ## animator(setup, "vorticity.mkv"; nupdate = 4),
    ## vtk_writer(setup; nupdate = 5, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 1),
);

# Solve unsteady problem
u, p, outputs =
    solve_unsteady(setup, u₀, p₀, tlims; Δt = T(0.001), processors, device);

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
## plot_streamfunction(setup, u)
nothing
