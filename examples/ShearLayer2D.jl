# # Shear layer case
#
# Shear layer test case.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using GLMakie

# Case name for saving results
name = "ShearLayer2D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = Inf)
# viscosity_model = KEpsilonModel{T}(; Re = Inf)
# viscosity_model = MixingLengthModel{T}(; Re = Inf)
# viscosity_model = SmagorinskyModel{T}(; Re = Inf)
# viscosity_model = QRModel{T}(; Re = Inf)

## Convection model
convection_model = NoRegConvectionModel()
# convection_model = C2ConvectionModel()
# convection_model = C4ConvectionModel()
# convection_model = LerayConvectionModel()

## Boundary conditions
u_bc(x, y, t) = 0.0
v_bc(x, y, t) = 0.0
bc = create_boundary_conditions(
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
x = stretched_grid(0.0, 2π, 40)
y = stretched_grid(0.0, 2π, 40)
grid = create_grid(x, y; bc, T, order4 = true);

# Plot grid
plot_grid(grid)

## Forcing parameters
bodyforce_u(x, y) = 0.0
bodyforce_v(x, y) = 0.0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

## Build setup and assemble operators
setup = Setup(; viscosity_model, convection_model, grid, force, bc)

## Pressure solver
pressure_solver = DirectPressureSolver(setup)
# pressure_solver = CGPressureSolver(setup)
# pressure_solver = FourierPressureSolver(setup)

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
    pressure_solver,
);


## Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem);


## Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 1, fieldname = :vorticity, type = contourf)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.1, processors, pressure_solver);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
