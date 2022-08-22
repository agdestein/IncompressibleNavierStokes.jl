# # Backward Facing Step case (BFS)
#
# This example considers a channel with periodic side boundaries, walls at the top and
# bottom, and a step at the left with a parabolic inflow. Initially the velocity is an
# extension of the inflow, but as time passes the velocity finds a new steady state.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using Revise
using Hardanger
using GLMakie
colorscheme!("Catppuccin")
set_theme!(makie(catppuccin()))

using IncompressibleNavierStokes
using GLMakie

# Case name for saving results
name = "BackwardFacingStep2D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 3000)
# viscosity_model = KEpsilonModel{T}(; Re = 2000)
# viscosity_model = MixingLengthModel{T}(; Re = 2000)
# viscosity_model = SmagorinskyModel{T}(; Re = 2000)
# viscosity_model = QRModel{T}(; Re = 2000)

## Convection model
convection_model = NoRegConvectionModel{T}()
# convection_model = C2ConvectionModel{T}()
# convection_model = C4ConvectionModel{T}()
# convection_model = LerayConvectionModel{T}()

## Boundary conditions
u_bc(x, y, t) = x ≈ 0 && y ≥ 0 ? 24y * (1 / 2 - y) : 0.0
v_bc(x, y, t) = 0.0
bc = create_boundary_conditions(
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:dirichlet, :pressure), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :symmetric), y = (:dirichlet, :dirichlet)),
    ),
    T,
)

## Grid
x = stretched_grid(0.0, 10.0, 300)
y = cosine_grid(-0.5, 0.5, 50)
grid = create_grid(x, y; bc, T);

plot_grid(grid)

## Forcing parameters
bodyforce_u(x, y) = 0.0
bodyforce_v(x, y) = 0.0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

## Build setup and assemble operators
setup = Setup{T,2}(; viscosity_model, convection_model, grid, force, bc);
build_operators!(setup);

## Pressure solver
pressure_solver = DirectPressureSolver{T}(setup)
# pressure_solver = CGPressureSolver{T}(setup)
# pressure_solver = FourierPressureSolver{T}(setup)

## Time interval
t_start, t_end = tlims = (0.0, 30.0)

## Initial conditions (extend inflow)
initial_velocity_u(x, y) = y ≥ 0.0 ? 24y * (1 / 2 - y) : 0.0
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
    pressure_solver,
);


## Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem);


## Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 5, fieldname = :vorticity, type = contour)
writer = VTKWriter(; nupdate = 20, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 10)
processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.002, processors, pressure_solver);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
