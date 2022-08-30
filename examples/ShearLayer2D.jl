# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Shear layer - 2D
#
# Shear layer example.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "ShearLayer2D"

# Viscosity model
viscosity_model = LaminarModel(; Re = Inf)

# A 2D grid is a Cartesian product of two vectors
n = 50
x = LinRange(0, 2π, n + 1)
y = LinRange(0, 2π, n + 1)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model);

## Time interval
t_start, t_end = tlims = (0.0, 8.0)

## Initial conditions
# we add 1 to u in order to make global momentum conservation less trivial
d = π / 15
e = 0.05
initial_velocity_u(x, y) = y ≤ π ? tanh((y - π / 2) / d) : tanh((3π / 2 - y) / d)
# initial_velocity_u(x, y) = 1.0 + (y ≤ π ? tanh((y - π / 2) / d) : tanh((3π / 2 - y) / d))
initial_velocity_v(x, y) = e * sin(x)
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
);

## Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = solve(problem);

## Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 1, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
# processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = solve(problem, RK44(); Δt = 0.1, processors, inplace = true);
#hide current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(V, p, t_end, setup, "output/solution")

# Plot tracers
plot_tracers(tracer)

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V, t_end)

# Plot vorticity
plot_vorticity(setup, V, t_end)

# Plot streamfunction
## plot_streamfunction(setup, V, t_end)
