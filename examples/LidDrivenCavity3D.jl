# # Lid-Driven Cavity case (LDC)
#
# This test case considers a box with a moving lid. The velocity is initially at rest. The
# solution should reach at steady state equilibrium after a certain time. The same steady
# state should be obtained when solving a `SteadyStateProblem`.

# LSP indexing solution
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using GLMakie

# Case name for saving results
name = "LDC"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 1000)
# viscosity_model = KEpsilonModel{T}(; Re = 1000)
# viscosity_model = MixingLengthModel{T}(; Re = 1000)
# viscosity_model = SmagorinskyModel{T}(; Re = 1000)
# viscosity_model = QRModel{T}(; Re = 1000)

## Convection model
convection_model = NoRegConvectionModel{T}()
# convection_model = C2ConvectionModel{T}()
# convection_model = C4ConvectionModel{T}()
# convection_model = LerayConvectionModel{T}()

## Nonuniform grid -- refine near walls
x = cosine_grid(0.0, 1.0, 25)
y = stretched_grid(0.0, 1.0, 25, 0.95)
z = stretched_grid(-0.2, 0.2, 10)
grid = create_grid(x, y, z; T);

plot_grid(grid)

## Solver settings
solver_settings = SolverSettings{T}(;
    pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
    # pressure_solver = CGPressureSolver{T}(),      # Pressure solver
    # pressure_solver = FourierPressureSolver{T}(), # Pressure solver
    p_add_solve = true,                             # Additional pressure solve for second order pressure
    abstol = 1e-10,                                 # Absolute accuracy
    reltol = 1e-14,                                 # Relative accuracy
    maxiter = 10,                                   # Maximum number of iterations
    # `:no`: Replace iteration matrix with I/Δt (no Jacobian)
    # `:approximate`: Build Jacobian once before iterations only
    # `:full`: Build Jacobian at each iteration
    newton_type = :approximate,
)

## Boundary conditions
u_bc(x, y, z, t, setup) = y ≈ setup.grid.ylims[2] ? 1.0 : 0.0
v_bc(x, y, z, t, setup) = zero(x)
w_bc(x, y, z, t, setup) = y ≈ setup.grid.ylims[2] ? 0.2 : 0.0
bc = create_boundary_conditions(
    u_bc,
    v_bc,
    w_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
        v = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
        w = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:periodic, :periodic),
        ),
    ),
    T,
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
t_start, t_end = tlims = (0.0, 10.0)

## Initial conditions
initial_velocity_u(x, y, z) = 0
initial_velocity_v(x, y, z) = 0
initial_velocity_w(x, y, z) = 0
initial_pressure(x, y, z) = 0
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
V, p = @time solve(problem; npicard = 5, maxiter = 15);


## Iteration processors
logger = Logger()
plotter = RealTimePlotter(; nupdate = 5, fieldname = :vorticity)
writer = VTKWriter(; nupdate = 5, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, writer, tracer]

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.01, processors)


## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])
