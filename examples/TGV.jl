# Taylor-Green vortex case (TG).

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using GLMakie

# Floating point type for simulations
T = Float64

# Spatial dimension
N = 3

# Case information
case = Case(;
    name = "TG",
    problem = UnsteadyProblem(),
    # problem = SteadyStateProblem(),
    regularization = "no",
)

# Viscosity model
model = LaminarModel{T}(; Re = 1000)
# model = KEpsilonModel{T}(; Re = 1000)
# model = MixingLengthModel{T}(; Re = 1000)
# model = SmagorinskyModel{T}(; Re = 1000)
# model = QRModel{T}(; Re = 1000)

# Grid parameters
grid = create_grid(
    T,
    N;
    Nx = 20,                         # Number of x-volumes
    Ny = 20,                         # Number of y-volumes
    Nz = 20,                         # Number of z-volumes
    xlims = (0, 2π),                  # Horizontal limits (left, right)
    ylims = (0, 2π),                  # Vertical limits (bottom, top)
    zlims = (0, 2π),                  # Depth limits (back, front)
    stretch = (1, 1, 1),              # Stretch factor (sx, sy[, sz])
)

# Time stepping
time = Time{T}(;
    t_start = 0,                       # Start time
    t_end = 10.0,                       # End time
    Δt = 0.05,                         # Timestep
    method = RK44(),                   # ODE method
    method_startup = RK44(),           # Startup method for methods that are not self-starting
    nstartup = 2,                      # Number of necessary Vₙ₋ᵢ (= method order)
    isadaptive = false,                # Adapt timestep every n_adapt_Δt iterations
    n_adapt_Δt = 1,                    # Number of iterations between timestep adjustment
    CFL = 0.5,                         # CFL number for adaptive methods
)

# Solver settings
solver_settings = SolverSettings{T}(;
    # pressure_solver = DirectPressureSolver{T}(),# Pressure solver
    # pressure_solver = CGPressureSolver{T}(; maxiter = 500, abstol = 1e-8),# Pressure solver
    pressure_solver = FourierPressureSolver{T}(),# Pressure solver
    p_initial = true,                # Calculate compatible IC for the pressure
    p_add_solve = true,              # Additional pressure solve to make it same order as velocity
    nonlinear_acc = 1e-10,           # Absolute accuracy
    nonlinear_relacc = 1e-14,        # Relative accuracy
    nonlinear_maxit = 10,            # Maximum number of iterations
    # "no": Replace iteration matrix with I/Δt (no Jacobian)
    # "approximate": Build Jacobian once before iterations only
    # "full": Build Jacobian at each iteration
    nonlinear_Newton = "full",
    Jacobian_type = "newton",        # Linearization: "picard", "newton"
    nonlinear_startingvalues = false,# Extrapolate values from last time step to get accurate initial guess (for unsteady problems only)
    nPicard = 6,                     # Number of Picard steps before switching to Newton when linearization is Newton (for steady problems only)
)

# Boundary conditions
bc_unsteady = false
bc_type = (;
    u = (;
        x = (:periodic, :periodic),
        y = (:periodic, :periodic),
        z = (:periodic, :periodic),
    ),
    v = (;
        x = (:periodic, :periodic),
        y = (:periodic, :periodic),
        z = (:periodic, :periodic),
    ),
    w = (;
        x = (:periodic, :periodic),
        y = (:periodic, :periodic),
        z = (:periodic, :periodic),
    ),
    k = (;
        x = (:periodic, :periodic),
        y = (:periodic, :periodic),
        z = (:periodic, :periodic),
    ),
    e = (;
        x = (:periodic, :periodic),
        y = (:periodic, :periodic),
        z = (:periodic, :periodic),
    ),
    ν = (;
        x = (:periodic, :periodic),
        y = (:periodic, :periodic),
        z = (:periodic, :periodic),
    ),
)
u_bc(x, y, z, t, setup) = zero(x)
v_bc(x, y, z, t, setup) = zero(x)
w_bc(x, y, z, t, setup) = zero(x)
dudt_bc(x, y, z, t, setup) = zero(x)
dvdt_bc(x, y, z, t, setup) = zero(x)
dwdt_bc(x, y, z, t, setup) = zero(x)
bc = create_boundary_conditions(
    T;
    bc_unsteady,
    bc_type,
    u_bc,
    v_bc,
    w_bc,
    dudt_bc,
    dvdt_bc,
    dwdt_bc,
)

# Initial conditions
initial_velocity_u(x, y, z) = sin(x)cos(y)cos(z)
initial_velocity_v(x, y, z) = -cos(x)sin(y)cos(z)
initial_velocity_w(x, y, z) = zero(z)
# initial_velocity_u(x, y, z) = -sinpi(x)cospi(y)cospi(z)
# initial_velocity_v(x, y, z) = 2cospi(x)sinpi(y)cospi(z)
# initial_velocity_w(x, y, z) = -cospi(x)cospi(y)sinpi(z)
# initial_pressure(x, y, z) = 1 / 4 * (cos(2π * x) + cos(2π * y) + cos(2π * z))
initial_pressure(x, y, z) = 1 / 4 * (cos(2x) + cos(2y) + cos(2z))
@pack! case = initial_velocity_u, initial_velocity_v, initial_velocity_w
@pack! case = initial_pressure

# Forcing parameters
bodyforce_u(x, y, z) = 0
bodyforce_v(x, y, z) = 0
bodyforce_w(x, y, z) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v, bodyforce_w)

# Iteration processors
logger = Logger()                        # Prints time step information
real_time_plotter = RealTimePlotter(;
    nupdate = 10,                        # Number of iterations between real time plots
    fieldname = :vorticity,              # Quantity for real time plotting
    # fieldname = :quiver,                 # Quantity for real time plotting
    # fieldname = :vorticity,              # Quantity for real time plotting
    # fieldname = :pressure,               # Quantity for real time plotting
    # fieldname = :streamfunction,         # Quantity for real time plotting
)
vtk_writer = VTKWriter(;
    nupdate = 1,                         # Number of iterations between VTK writings
    dir = "output/$(case.name)",         # Output directory
    filename = "solution",               # Output file name (without extension)
)
tracer = QuantityTracer(; nupdate = 1)   # Stores tracer data
processors = [logger, vtk_writer, tracer]

# Final setup
setup = Setup{T,N}(;
    case,
    model,
    grid,
    force,
    time,
    solver_settings,
    processors,
    bc,
)

## Prepare
build_operators!(setup);
V₀, p₀, t₀ = create_initial_conditions(setup);

## Solve problem
problem = setup.case.problem;
@time V, p = solve(problem, setup, V₀, p₀);

## Plot tracers
plot_tracers(setup)

## Post-process
plot_pressure(setup, p)
plot_vorticity(setup, V, setup.time.t_end)
plot_streamfunction(setup, V, setup.time.t_end)
