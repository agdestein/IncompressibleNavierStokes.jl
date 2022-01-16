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
name = "BFS"

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
Nx = 100                             # Number of x-volumes
Ny = 16                              # Number of y-volumes
Nz = 8                               # Number of z-volumes
grid = create_grid(
    T,
    Nx,
    Ny,
    Nz;
    xlims = (0, 10),                 # Horizontal limits (left, right)
    ylims = (-0.5, 0.5),             # Vertical limits (bottom, top)
    zlims = (-0.25, 0.25),           # Depth limits (back, front)
    stretch = (1, 1, 1),             # Stretch factor (sx, sy[, sz])
)

## Solver settings
solver_settings = SolverSettings{T}(;
    pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
    # pressure_solver = CGPressureSolver{T}(),      # Pressure solver
    # pressure_solver = FourierPressureSolver{T}(), # Pressure solver
    p_add_solve = true,                             # Additional pressure solve to make it same order as velocity
    abstol = 1e-10,                                 # Absolute accuracy
    reltol = 1e-14,                                 # Relative accuracy
    maxiter = 10,                                   # Maximum number of iterations
    # :no: Replace iteration matrix with I/Δt (no Jacobian)
    # :approximate: Build Jacobian once before iterations only
    # :full: Build Jacobian at each iteration
    newton_type = :full,
)

## Boundary conditions
u_bc(x, y, z, t, setup) = x ≈ setup.grid.xlims[1] && y ≥ 0 ? 24y * (1 // 2 - y) : zero(x)
v_bc(x, y, z, t, setup) = zero(x)
w_bc(x, y, z, t, setup) = zero(x)
bc = create_boundary_conditions(
    T,
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
        k = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
        e = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
        ν = (;
            x = (:symmetric, :symmetric),
            y = (:symmetric, :symmetric),
            z = (:symmetric, :symmetric),
        ),
    ),
)

## Forcing parameters
bodyforce_u(x, y, z) = 0
bodyforce_v(x, y, z) = 0
bodyforce_w(x, y, z) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v, bodyforce_w)

## Build setup and assemble operators
setup = Setup{T,3}(; viscosity_model, convection_model, grid, force, solver_settings, bc);
build_operators!(setup);

## Time interval
t_start, t_end = tlims = (0.0, 20.0)

## Initial conditions (extend inflow)
initial_velocity_u(x, y, z) = y ≥ 0 ? 24y * (1 // 2 - y) : zero(y)
initial_velocity_v(x, y, z) = zero(x)
initial_velocity_w(x, y, z) = zero(x)
initial_pressure(x, y, z) = zero(x)
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
real_time_plotter = RealTimePlotter(; nupdate = 10, fieldname = :velocity)
vtk_writer = VTKWriter(; nupdate = 20, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, real_time_plotter, vtk_writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.005, processors);


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
