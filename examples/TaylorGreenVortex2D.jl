# # Taylor-Green vortex case (2D).
#
# This test case considers the Taylor-Green vortex.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DifferentiableNavierStokes.jl")
    using .DifferentiableNavierStokes
end

using Revise
using Hardanger
using GLMakie
colorscheme!("GruvboxDark")
set_theme!(makie(gruvbox()))

using DifferentiableNavierStokes
using GLMakie

# Case name for saving results
name = "TaylorGreenVortex2D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 100)
# viscosity_model = MixingLengthModel{T}(; Re = 1000)
# viscosity_model = SmagorinskyModel{T}(; Re = 1000)
# viscosity_model = QRModel{T}(; Re = 1000)

## Grid
x = stretched_grid(0, 2, 100)
y = stretched_grid(0, 2, 100)
grid = create_grid(x, y; T);

plot_grid(grid)

## Forcing parameters
bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

## Pressure solver
# pressure_solver = DirectPressureSolver{T}()
# pressure_solver = CGPressureSolver{T}(; maxiter = 500, abstol = 1e-8)
pressure_solver = FourierPressureSolver{T}()

## Build setup and assemble operators
setup = Setup{T,2}(; viscosity_model, grid, force, pressure_solver);
build_operators!(setup);

## Time interval
t_start, t_end = tlims = (0.0, 5.0)

## Initial conditions
initial_velocity_u(x, y) = -sinpi(x)cospi(y)
initial_velocity_v(x, y) = cospi(x)sinpi(y)
initial_pressure(x, y) = 1 / 4 * (cospi(2x) + cospi(2y))
V₀, p₀ = create_initial_conditions(
    setup;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
);


## Iteration processors
logger = Logger()
plotter = RealTimePlotter(; nupdate = 10, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, writer, tracer]
# processors = []

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.01, processors)


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
