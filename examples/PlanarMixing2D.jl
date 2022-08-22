# # Planar mixing case
#
# Planar mixing test case.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using GLMakie

# Case name for saving results
name = "PlanarMixing2D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 500)
# viscosity_model = KEpsilonModel{T}(; Re = 500)
# viscosity_model = MixingLengthModel{T}(; Re = 500)
# viscosity_model = SmagorinskyModel{T}(; Re = 500)
# viscosity_model = QRModel{T}(; Re = 500)

## Convection model
convection_model = NoRegConvectionModel{T}()
# convection_model = C2ConvectionModel{T}()
# convection_model = C4ConvectionModel{T}()
# convection_model = LerayConvectionModel{T}()

## Boundary conditions ΔU = 1.0
ΔU = 1.0
Ū = 1.0
ϵ = (0.082Ū, 0.012Ū)
n = (0.4π, 0.3π)
ω = (0.22, 0.11)
u_bc(x, y, t) = 1.0 + ΔU / 2 * tanh(2y) + sum(@. ϵ * (1 - tanh(y / 2)^2) * cos(n * y) * sin(ω * t))
v_bc(x, y, t) = 0.0
dudt_bc(x, y, t) = sum(@. ϵ * (1 - tanh(y / 2)^2) * cos(n * y) * ω * cos(ω * t))
dvdt_bc(x, y, t) = 0.0
bc = create_boundary_conditions(
    u_bc,
    v_bc;
    dudt_bc,
    dvdt_bc,
    bc_unsteady = true,
    bc_type = (;
        u = (; x = (:dirichlet, :pressure), y = (:symmetric, :symmetric)),
        v = (; x = (:dirichlet, :symmetric), y = (:pressure, :pressure)),
    ),
    T,
)

## Grid
x = stretched_grid(0.0, 256.0, 1024)
y = stretched_grid(-32.0, 32.0, 256)
grid = create_grid(x, y; bc, T, order4 = false);

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
t_start, t_end = tlims = (0.0, 300.0)

## Initial conditions
initial_velocity_u(x, y) = u_bc(x, y, 0.0)
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
plotter = RealTimePlotter(; nupdate = 10, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 10)
processors = [logger, plotter, writer, tracer]
# , lims = (-0.8, 0.02)

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44P2(); Δt = 0.1, processors, pressure_solver);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2]);
plot_streamfunction(setup, V, tlims[2])
