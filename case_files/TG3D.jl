"""
    TG3D()

Create setup for Taylor-Green vortex case (TG).
"""
function TG3D()
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

    # Physical properties
    fluid = Fluid{T}(;
        Re = 1000,                        # Reynolds number
        U1 = 1,                           # Velocity scales
        U2 = 1,                           # Velocity scales
        d_layer = 1,                      # Thickness of layer
    )

    # Viscosity model
    model = LaminarModel{T}()
    # model = KEpsilonModel{T}()
    # model = MixingLengthModel{T}()
    # model = SmagorinskyModel{T}()
    # model = QRModel{T}()

    # Grid parameters
    grid = create_grid(
        T,
        N;
        Nx = 40,                         # Number of x-volumes
        Ny = 30,                         # Number of y-volumes
        Nz = 20,                         # Number of z-volumes
        xlims = (0, 2π),                  # Horizontal limits (left, right)
        ylims = (0, 2π),                  # Vertical limits (bottom, top)
        zlims = (0, 2π),                  # Depth limits (back, front)
        stretch = (1, 1, 1),              # Stretch factor (sx, sy[, sz])
    )

    # Discretization parameters
    discretization = Operators{T}(;
        order4 = false,                   # Use 4th order in space (otherwise 2nd order)
        α = 81,                           # Richardson extrapolation factor = 3^4
        β = 9 / 8,                        # Interpolation factor
    )

    # Rom parameters
    rom = ROM(;
        use_rom = false,                  # Use reduced order model
        rom_type = "POD",                 # "POD", "Fourier"
        M = 10,                           # Number of ROM velocity modes
        Mp = 10,                          # Number of ROM pressure modes
        precompute_convection = true,     # Precomputed convection matrices
        precompute_diffusion = true,      # Precomputed diffusion matrices
        precompute_force = true,          # Precomputed forcing term
        t_snapshots = 0,                  # Snapshots
        Δt_snapshots = false,             # Gap between snapshots
        mom_cons = false,                 # Momentum conserving SVD
        # ROM boundary constitions:
        # 0: homogeneous (no-slip = periodic)
        # 1: non-homogeneous = time-independent
        # 2: non-homogeneous = time-dependent
        rom_bc = 0,
        weighted_norm = true,             # Using finite volumes as weights
        pressure_recovery = false,        # Compute pressure with PPE-ROM
        pressure_precompute = 0,          # Recover pressure with FOM (0) or ROM (1)
        subtract_pressure_mean = false,   # Subtract pressure mean from snapshots
        process_iteration_FOM = true,     # FOM divergence, residuals, and kinetic energy
        basis_type = "default",           # "default", "svd", "direct", "snapshot"
    )

    # Immersed boundary method
    ibm = IBM(; use_ibm = false)      # Use immersed boundary method

    # Time stepping
    time = Time{T}(;
        t_start = 0,                       # Start time
        t_end = 1,                         # End time
        Δt = 0.01,                         # Timestep
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
    initial_pressure(x, y, z) = 1 / 4 * (cos(2π * x) + cos(2π * y) + cos(2π * z))
    @pack! case = initial_velocity_u, initial_velocity_v, initial_pressure

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
        nupdate = 10,                        # Number of iterations between VTK writings
        dir = "output/$(case.name)",                # Output directory
        filename = "solution",               # Output file name (without extension)
    )
    processors = [logger, real_time_plotter, vtk_writer]

    # Final setup
    Setup{T,N}(;
        case,
        fluid,
        model,
        grid,
        discretization,
        force,
        rom,
        ibm,
        time,
        solver_settings,
        processors,
        bc,
    )
end
