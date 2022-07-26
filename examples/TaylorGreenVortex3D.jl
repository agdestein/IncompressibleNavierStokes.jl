# # Taylor-Green vortex case (3D).
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
name = "TaylorGreenVortex3D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 2000)
# viscosity_model = MixingLengthModel{T}(; Re = 1000)
# viscosity_model = SmagorinskyModel{T}(; Re = 1000)
# viscosity_model = QRModel{T}(; Re = 1000)

## Grid
x = stretched_grid(0, 2, 20)
y = stretched_grid(0, 2, 20)
z = stretched_grid(0, 2, 20)
grid = create_grid(x, y, z; T);

plot_grid(grid)

## Forcing parameters
bodyforce_u(x, y, z) = 0.0
bodyforce_v(x, y, z) = 0.0
bodyforce_w(x, y, z) = 0.0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v, bodyforce_w)

## Pressure solver
# pressure_solver = DirectPressureSolver{T}()
# pressure_solver = CGPressureSolver{T}(; maxiter = 500, abstol = 1e-8)
pressure_solver = FourierPressureSolver{T}()

## Build setup and assemble operators
setup = Setup{T,3}(; viscosity_model,  grid, force, pressure_solver);
@time build_operators!(setup);

## Time interval
t_start, t_end = tlims = (0.0, 10.0)

## Initial conditions
initial_velocity_u(x, y, z) = sinpi(x)cospi(y)cospi(z)
initial_velocity_v(x, y, z) = -cospi(x)sinpi(y)cospi(z)
initial_velocity_w(x, y, z) = 0.0
initial_pressure(x, y, z) = 1 / 4 * (cospi(2x) + cospi(2y) + cospi(2z))
V₀, p₀ = create_initial_conditions(
    setup;
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure,
);


## Iteration processors
logger = Logger()
plotter = RealTimePlotter(; nupdate = 10, fieldname = :velocity)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.01, processors)


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p; alpha = 0.05)
plot_velocity(setup, V, t_end; alpha = 0.05)
plot_vorticity(setup, V, tlims[2]; alpha = 0.05)
plot_streamfunction(setup, V, tlims[2])
