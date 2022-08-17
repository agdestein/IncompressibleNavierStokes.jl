# # Shear layer case
#
# Shear layer test case.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DifferentiableNavierStokes.jl")
    using .DifferentiableNavierStokes
end

using DifferentiableNavierStokes
using GLMakie

# Case name for saving results
name = "ShearLayer2D"

# Floating point type for simulations
T = Float64

## Grid
x = stretched_grid(0.0, 2π, 200)
y = stretched_grid(0.0, 2π, 200)
grid = create_grid(x, y; T);

# Plot grid
plot_grid(grid)

## Assemble operators
operators = build_operators(grid);

## Body force
bodyforce_u(x, y) = 0.0
bodyforce_v(x, y) = 0.0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)
DifferentiableNavierStokes.build_force!(force, grid)

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = Inf)
# viscosity_model = MixingLengthModel{T}(; Re = Inf)
# viscosity_model = SmagorinskyModel{T}(; Re = Inf)
# viscosity_model = QRModel{T}(; Re = Inf)

## Setup
setup = Setup(; grid, operators, viscosity_model, force);

## Pressure solver
# pressure_solver = DirectPressureSolver{T}()
# pressure_solver = CGPressureSolver{T}()
pressure_solver = FourierPressureSolver(setup)

## Initial conditions
# we add 1 to u in order to make global momentum conservation less trivial
d = π / 15
e = 0.05
initial_velocity_u(x, y) = y ≤ π ? tanh((y - π / 2) / d) : tanh((3π / 2 - y) / d)
# initial_velocity_u(x, y) = 1.0 + (y ≤ π ? tanh((y - π / 2) / d) : tanh((3π / 2 - y) / d))
initial_velocity_v(x, y) = e * sin(x)
# initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup;
    initial_velocity_u,
    initial_velocity_v,
    # initial_pressure,
    pressure_solver,
);
V, p = V₀, p₀

## Time interval
t_start, t_end = tlims = (0.0, 8.0)

## Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 1, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
# processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
# problem = UnsteadyProblem(setup, V, p, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.001, processors, pressure_solver);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
