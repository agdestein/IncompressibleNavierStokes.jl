# # Backward Facing Step case (BFS)
#
# This example considers a channel with periodic side boundaries, walls at the top and
# bottom, and a step at the left with a parabolic inflow. Initially the velocity is an
# extension of the inflow, but as time passes the velocity finds a new steady state.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using GLMakie

# Case name for saving results
name = "BackwardFacingStep3D"

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

## Grid
x = stretched_grid(0, 10, 160)
y = stretched_grid(-0.5, 0.5, 16)
z = stretched_grid(-0.25, 0.25, 8)
grid = create_grid(x, y, z; T);

plot_grid(grid)

## Boundary conditions
u_bc(x, y, z, t) = x ≈ grid.xlims[1] && y ≥ 0 ? 24y * (1 / 2 - y) : 0.0
v_bc(x, y, z, t) = 0.0
w_bc(x, y, z, t) = 0.0
bc = create_boundary_conditions(
    u_bc,
    v_bc,
    w_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (;
            x = (:dirichlet, :pressure),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
        v = (;
            x = (:dirichlet, :symmetric),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
        w = (;
            x = (:dirichlet, :symmetric),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
    ),
    T,
)

## Forcing parameters
bodyforce_u(x, y, z) = 0.0
bodyforce_v(x, y, z) = 0.0
bodyforce_w(x, y, z) = 0.0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v, bodyforce_w)

## Pressure solver
pressure_solver = DirectPressureSolver{T}()
# pressure_solver = CGPressureSolver{T}()
# pressure_solver = FourierPressureSolver{T}()

## Build setup and assemble operators
setup = Setup{T,3}(; viscosity_model, convection_model, grid, force, pressure_solver, bc);
build_operators!(setup);

## Time interval
t_start, t_end = tlims = (0.0, 25.0)

## Initial conditions (extend inflow)
initial_velocity_u(x, y, z) = y ≥ 0 ? 24y * (1 / 2 - y) : 0.0
initial_velocity_v(x, y, z) = 0.0
initial_velocity_w(x, y, z) = 0.0
initial_pressure(x, y, z) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure,
);


## Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem);


## Iteration processors
logger = Logger(; nupdate = 10)
plotter = RealTimePlotter(; nupdate = 50, fieldname = :velocity)
writer = VTKWriter(; nupdate = 20, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 25)
processors = [logger, plotter, writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.01, processors);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
