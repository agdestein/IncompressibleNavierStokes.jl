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
convection_model = NoRegConvectionModel{T}()
# convection_model = C2ConvectionModel{T}()
# convection_model = C4ConvectionModel{T}()
# convection_model = LerayConvectionModel{T}()

## Grid
x = stretched_grid(0.0, 2π, 40)
y = stretched_grid(0.0, 2π, 40)
grid = create_grid(x, y; T, order4 = true);

plot_grid(grid)

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

## Forcing parameters
bodyforce_u(x, y) = 0.0
bodyforce_v(x, y) = 0.0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

## Pressure solver
pressure_solver = DirectPressureSolver{T}()
# pressure_solver = CGPressureSolver{T}()
# pressure_solver = FourierPressureSolver{T}()

## Build setup and assemble operators
setup = Setup{T,2}(; viscosity_model, convection_model, grid, force, pressure_solver, bc);
build_operators!(setup);

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
V, p = @time solve(problem);


## Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 1, fieldname = :vorticity, type = contourf)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.1, processors);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
