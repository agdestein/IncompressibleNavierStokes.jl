# # Unsteady actuator case
#
# An unsteady inlet velocity profile at encounters a wind turbine blade in a wall-less
# domain. The blade is modeled as a uniform body force on a thin rectangle.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using GLMakie

# Case name for saving results
name = "Actuator3D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 100)
# viscosity_model = KEpsilonModel{T}(; Re = 100)
# viscosity_model = MixingLengthModel{T}(; Re = 100)
# viscosity_model = SmagorinskyModel{T}(; Re = 100)
# viscosity_model = QRModel{T}(; Re = 100)

## Convection model
convection_model = NoRegConvectionModel{T}() # convection_model = C2ConvectionModel{T}()
# convection_model = C4ConvectionModel{T}()
# convection_model = LerayConvectionModel{T}()

## Boundary conditions
f = 0.5
u_bc(x, y, z, t) = x ≈ grid.xlims[1] ? cos(π / 6 * sin(f * t)) : 0.0
v_bc(x, y, z, t) = x ≈ grid.xlims[1] ? sin(π / 6 * sin(f * t)) : 0.0
w_bc(x, y, z, t) = 0.0
dudt_bc(x, y, z, t) =
    x ≈ grid.xlims[1] ? -π / 6 * f * cos(f * t) * sin(π / 6 * sin(f * t)) : 0.0
dvdt_bc(x, y, z, t) =
    x ≈ grid.xlims[1] ? π / 6 * f * cos(f * t) * cos(π / 6 * sin(f * t)) : 0.0
dwdt_bc(x, y, z, t) = 0.0
bc = create_boundary_conditions(
    u_bc,
    v_bc,
    w_bc;
    dudt_bc,
    dvdt_bc,
    dwdt_bc,
    bc_unsteady = true,
    bc_type = (;
        u = (;
            x = (:dirichlet, :pressure),
            y = (:symmetric, :symmetric),
            z = (:symmetric, :symmetric),
        ),
        v = (;
            x = (:dirichlet, :symmetric),
            y = (:pressure, :pressure),
            z = (:symmetric, :symmetric),
        ),
        w = (;
            x = (:dirichlet, :symmetric),
            y = (:symmetric, :symmetric),
            z = (:pressure, :pressure),
        ),
    ),
    T,
)

## Grid
x = stretched_grid(0.0, 6.0, 30)
y = stretched_grid(-2.0, 2.0, 40)
z = stretched_grid(-2.0, 2.0, 40)
grid = create_grid(x, y, z; bc, T);

plot_grid(grid)

## Forcing parameters
cx, cy, cz = 2.0, 0.0, 0.0 # Disk center
D = 1.0 # Disk diameter
δ = 0.11 # Disk thickness
Cₜ = 0.01 # Thrust coefficient
inside(x, y, z) = abs(x - cx) ≤ δ / 2 && (y - cy)^2 + (z - cz)^2 ≤ (D / 2)^2
bodyforce_u(x, y, z) = -Cₜ * inside(x, y, z)
bodyforce_v(x, y, z) = 0.0
bodyforce_w(x, y, z) = 0.0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, bodyforce_w, grid)

## Build setup and assemble operators
setup = Setup(; viscosity_model, convection_model, grid, force, bc)

## Pressure solver
pressure_solver = DirectPressureSolver(setup)
# pressure_solver = CGPressureSolver(setup)
# pressure_solver = FourierPressureSolver(setup)

## Time interval
t_start, t_end = tlims = (0.0, 4π)

## Initial conditions (extend inflow)
initial_velocity_u(x, y, z) = 1.0
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
    pressure_solver,
);


## Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem);


## Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 5, fieldname = :vorticity)
writer = VTKWriter(; nupdate = 2, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44P2(); Δt = 4π / 200, processors, pressure_solver);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
plot_force(setup, setup.force.F, t_end)
