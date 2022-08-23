# # Taylor-Green vortex case (TG).
#
# This test case considers the Taylor-Green vortex.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using GLMakie

# Case name for saving results
name = "TaylorGreenVortex2D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 2000)
# viscosity_model = KEpsilonModel{T}(; Re = 1000)
# viscosity_model = MixingLengthModel{T}(; Re = 1000)
# viscosity_model = SmagorinskyModel{T}(; Re = 1000)
# viscosity_model = QRModel{T}(; Re = 1000)

## Convection model
convection_model = NoRegConvectionModel()
# convection_model = C2ConvectionModel()
# convection_model = C4ConvectionModel()
# convection_model = LerayConvectionModel()

## Boundary conditions
u_bc(x, y, t) = zero(x)
v_bc(x, y, t) = zero(x)
bc = BC(
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        v = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
    ),
    T,
)

## Grid
x = stretched_grid(0, 2π, 20)
y = stretched_grid(0, 2π, 20)
grid = Grid(x, y; bc, T);

plot_grid(grid)

## Forcing parameters
bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

## Build setup and assemble operators
setup = Setup(; viscosity_model, convection_model, grid, force, bc)

## Pressure solver
# pressure_solver = DirectPressureSolver(setup)
# pressure_solver = CGPressureSolver(setup)
pressure_solver = FourierPressureSolver(setup)

## Time interval
t_start, t_end = tlims = (0.0, 50.0)

## Initial conditions
initial_velocity_u(x, y) = -sin(x)cos(y)
initial_velocity_v(x, y) = cos(x)sin(y)
initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
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
V, p = @time solve(problem; npicard = 2);


## Iteration processors
logger = Logger()
plotter = RealTimePlotter(; nupdate = 10, fieldname = :vorticity)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.01, processors, pressure_solver)


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
