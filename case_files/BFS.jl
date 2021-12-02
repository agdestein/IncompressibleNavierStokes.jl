"""
    BFS()

Create setup for unsteady Backward Facing Step case (BFS).
"""
function BFS()
    # Floating point type for simulations
    T = Float64

    # Spatial dimension
    N = 2

    # Case information
    case = Case(;
        name = "BFS",
        problem = UnsteadyProblem(),
        # problem = SteadyStateProblem(),
        regularization = "no",
    )

    # Physical properties
    fluid = Fluid{T}(;
        Re = 1000,                # Reynolds number
        U1 = 1,                   # Velocity scales
        U2 = 1,                   # Velocity scales
        d_layer = 1,              # Thickness of layer
    )

    # Viscosity model
    model = LaminarModel{T}()
    # model = KEpsilonModel{T}()
    # model = MixingLengthModel{T}()
    # model = SmagorinskyModel{T}()
    # model = QRModel{T}()

    # Grid parameters
    grid = create_grid(T, N;
        Nx = 400,                        # Number of x-volumes
        Ny = 40,                         # Number of y-volumes
        xlims = (0, 10),                  # Horizontal limits (left, right)
        ylims = (-0.5, 0.5),              # Vertical limits (bottom, top)
        stretch = (1, 1),                 # Stretch factor (sx, sy[, sz])
    )

    # Discrete operators
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
    ibm = IBM(; use_ibm = false)     # Use immersed boundary method

    # Time stepping
    time = Time{T}(;
        t_start = 0,                 # Start time
        t_end = 20,                  # End time
        Δt = 0.02,                   # Timestep
        method = RK44(),             # ODE method
        method_startup = RK44(),     # Startup method for methods that are not self-starting
        nstartup = 2,                # Number of necessary Vₙ₋ᵢ (= method order)
        isadaptive = false,          # Adapt timestep every n_adapt_Δt iterations
        n_adapt_Δt = 1,              # Number of iterations between timestep adjustment
        CFL = 0.5,                   # CFL number for adaptive methods
    )

    # Solver settings
    solver_settings = SolverSettings{T}(;
        pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
        # pressure_solver = CGPressureSolver{T}(),      # Pressure solver
        # pressure_solver = FourierPressureSolver{T}(), # Pressure solver
        p_initial = true,                               # Calculate compatible IC for the pressure
        p_add_solve = true,                             # Additional pressure solve to make it same order as velocity
        nonlinear_acc = 1e-10,                          # Absolute accuracy
        nonlinear_relacc = 1e-14,                       # Relative accuracy
        nonlinear_maxit = 10,                           # Maximum number of iterations
        # "no": Replace iteration matrix with I/Δt (no Jacobian)
        # "approximate": Build Jacobian once before iterations only
        # "full": Build Jacobian at each iteration
        nonlinear_Newton = "full",
        Jacobian_type = "newton",        # Linearization: "picard", "newton"
        nonlinear_startingvalues = false, # Extrapolate values from last time step
        nPicard = 2, # Number of Picard steps before Newton
    )

    # Boundary conditions
    bc_unsteady = false
    bc_type = (;
        u = (; x = (:dirichlet, :pressure), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :symmetric), y = (:dirichlet, :dirichlet)),
        k = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        e = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ν = (; x = (:symmetric, :symmetric), y = (:symmetric, :symmetric)),
    )
    u_bc(x, y, t, setup) = x ≈ setup.grid.xlims[1] && y ≥ 0 ? 24y * (1 // 2 - y) : zero(y)
    v_bc(x, y, t, setup) = 0
    bc = create_boundary_conditions(T; bc_unsteady, bc_type, u_bc, v_bc)

    # Initial conditions (extend inflow)
    initial_velocity_u(x, y) = y ≥ 0 ? 24y * (1 // 2 - y) : zero(y)
    initial_velocity_v(x, y) = 0
    initial_pressure(x, y) = 0
    @pack! case = initial_velocity_u, initial_velocity_v, initial_pressure

    # Forcing parameters
    bodyforce_u(x, y) = 0
    bodyforce_v(x, y) = 0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

    # Iteration processors
    logger = Logger()                        # Prints time step information
    real_time_plotter = RealTimePlotter(;
        nupdate = 5,                         # Number of iterations between real time plots
        fieldname = :vorticity,              # Quantity for real time plotting
        # fieldname = :quiver,                 # Quantity for real time plotting
        # fieldname = :vorticity,              # Quantity for real time plotting
        # fieldname = :pressure,               # Quantity for real time plotting
        # fieldname = :streamfunction,         # Quantity for real time plotting
    )
    vtk_writer = VTKWriter(;
        nupdate = 10,                        # Number of iterations between VTK writings
        dir = "output/$(case.name)",         # Output directory
        filename = "solution",               # Output file name (without extension)
    )
    processors = [logger, real_time_plotter, vtk_writer]
    # processors = [logger, real_time_plotter]

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
