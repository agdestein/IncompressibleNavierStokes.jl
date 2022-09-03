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

# Viscosity model
viscosity_model = LaminarModel(; Re = 2000.0)

# A 3D grid is a Cartesian product of three vectors
n = 20
x = LinRange(0, 2π, n + 1)
y = LinRange(0, 2π, n + 1)
z = LinRange(0, 2π, n + 1)
plot_grid(x, y, z)

# Build setup and assemble operators
setup = Setup(x, y, z; viscosity_model);

# Since the grid is uniform and identical for x, y, and z, we may use a
# specialized Fourier pressure solver
pressure_solver = FourierPressureSolver(setup)

# Time interval
t_start, t_end = tlims = (0.0, 5.0)

# Initial conditions
initial_velocity_u(x, y, z) = sin(x)cos(y)cos(z)
initial_velocity_v(x, y, z) = -cos(x)sin(y)cos(z)
initial_velocity_w(x, y, z) = 0.0
initial_pressure(x, y, z) = 1 / 4 * (cos(2x) + cos(2y) + cos(2z))
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure,
    pressure_solver,
);

# Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = solve(problem; npicard = 6)

# Iteration processors
logger = Logger()
plotter = RealTimePlotter(; nupdate = 10, fieldname = :vorticity, type = contour)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
## processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

# Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = solve(problem, RK44(); Δt = 0.01, processors, pressure_solver, inplace = true)
#md current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(V, p, t_end, setup, "output/solution")

# Plot tracers
plot_tracers(tracer)

# Plot pressure
plot_pressure(setup, p; alpha = 0.05)

# Plot velocity
plot_velocity(setup, V, t_end; alpha = 0.05)

# Plot vorticity
plot_vorticity(setup, V, t_end; alpha = 0.05)

# Plot streamfunction
## plot_streamfunction(setup, V, t_end)
