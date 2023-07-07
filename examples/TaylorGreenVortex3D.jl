# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Taylor-Green vortex - 3D
#
# In this example we consider the Taylor-Green vortex.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "TaylorGreenVortex3D"

# Floating point precision
T = Float32

# For CPU
device = identity

# For GPU (note that `cu` converts to `Float32`)
## using CUDA
## device = cu

# Viscosity model
viscosity_model = LaminarModel(; Re = T(2_000))

# A 3D grid is a Cartesian product of three vectors
n = 32
lims = (T(0), T(2π))
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
z = LinRange(lims..., n + 1)
plot_grid(x, y, z)

# Build setup and assemble operators
setup = Setup(x, y, z; viscosity_model);

# Since the grid is uniform and identical for x, y, and z, we may use a
# specialized Fourier pressure solver
pressure_solver = FourierPressureSolver(setup)

# Initial conditions
initial_velocity_u(x, y, z) = sin(x)cos(y)cos(z)
initial_velocity_v(x, y, z) = -cos(x)sin(y)cos(z)
initial_velocity_w(x, y, z) = zero(x)
initial_pressure(x, y, z) = 1 // 4 * (cos(2x) + cos(2y) + cos(2z))
V₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    T(0);
    initial_pressure,
    pressure_solver,
);

# Solve steady state problem
V, p = solve_steady_state(setup, V₀, p₀; npicard = 6)

# Iteration processors
processors = (
    field_plotter(device(setup); nupdate = 5),
    ## energy_history_plotter(device(setup); nupdate = 1),
    ## energy_spectrum_plotter(device(setup); nupdate = 100),
    ## animator(device(setup), "vorticity.mp4"; nupdate = 4),
    ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 10),
);

# Time interval
t_start, t_end = tlims = (T(0), T(5))

# Solve unsteady problem
V, p, outputs = solve_unsteady(
    setup,
    V₀,
    p₀,
    tlims;
    Δt = T(0.01),
    processors,
    pressure_solver,
    inplace = true,
    device,
);
#md current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(setup, setup, V, p, "output/solution")

# Plot pressure
plot_pressure(setup, p; levels = 3, alpha = 0.05)

# Plot velocity
plot_velocity(setup, V, t_end; levels = 3, alpha = 0.05)

# Plot vorticity
plot_vorticity(setup, V, t_end; levels = 5, alpha = 0.05)

# Plot streamfunction
## plot_streamfunction(setup, V, t_end)
