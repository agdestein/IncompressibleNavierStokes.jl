# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Taylor-Green vortex - 2D
#
# In this example we consider the Taylor-Green vortex.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

using CUDA
#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "TaylorGreenVortex2D"

# Floating point type
T = Float32

# Viscosity model
viscosity_model = LaminarModel(; Re = T(2_000))

# A 2D grid is a Cartesian product of two vectors
n = 128
lims = (T(0), T(2π))
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model);

# Since the grid is uniform and identical for x and y, we may use a specialized
# Fourier pressure solver
pressure_solver = FourierPressureSolver(setup)

# Time interval
t_start, t_end = tlims = (T(0), T(1))

# Initial conditions
initial_velocity_u(x, y) = -sin(x)cos(y)
initial_velocity_v(x, y) = cos(x)sin(y)
initial_pressure(x, y) = 1 // 4 * (cos(2x) + cos(2y))
V₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity_u,
    initial_velocity_v,
    t_start;
    initial_pressure,
    pressure_solver,
);

# Solve steady state problem
V, p = solve_steady_state(setup, V₀, p₀; npicard = 2);

# Iteration processors
processors = (
    field_plotter(setup; nupdate = 1),
    # energy_history_plotter(setup; nupdate = 1),
    # energy_spectrum_plotter(setup; nupdate = 1),
    step_logger(; nupdate = 1),
    # vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
);

# Solve unsteady problem
@time V, p, outputs = solve_unsteady(
    setup,
    V₀,
    p₀,
    tlims;
    Δt = T(0.01),
    processors,
    pressure_solver,
    inplace = true,
);
#md current_figure()

cusetup = cu(setup);

# Iteration processors
processors = (
    field_plotter(cusetup; nupdate = 1),
    # energy_history_plotter(setup; nupdate = 1),
    # energy_spectrum_plotter(setup; nupdate = 1),
    step_logger(; nupdate = 1),
    # vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
);

# Solve unsteady problem
@time V, p, outputs = solve_unsteady(
    cusetup,
    cu(V₀),
    cu(p₀),
    tlims;
    Δt = T(0.01),
    processors,
    pressure_solver = FourierPressureSolver(cusetup),
    inplace = true,
    bc_vectors = cu(get_bc_vectors(setup, t_start)),
);
#md current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(setup, V, p, t_end, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V, t_end)

# Plot vorticity
plot_vorticity(setup, V, t_end)

# Plot streamfunction
## plot_streamfunction(setup, V, t_end)
