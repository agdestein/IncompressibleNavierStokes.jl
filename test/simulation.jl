# Run a typical simulation
@testset "Simulation" begin
    # Floating point type for simulations
    T = Float64

    # Spatial dimension
    N = 2

    # Case information
    name = "LDC"
    problem = UnsteadyProblem()
    # problem = UnsteadyProblem()
    regularization = "no"
    case = Case(; name, problem, regularization)

    # Physical properties
    Re = 1000                         # Reynolds number
    U1 = 1                            # Velocity scales
    U2 = 1                            # Velocity scales
    d_layer = 1                       # Thickness of layer
    fluid = Fluid{T}(; Re, U1, U2, d_layer)

    # Viscosity model
    model = LaminarModel{T}()
    # model = KEpsilonModel{T}()
    # model = MixingLengthModel{T}()
    # model = SmagorinskyModel{T}()
    # model = QRModel{T}()

    # Grid parameters
    Nx = 80                           # Number of x-volumes
    Ny = 80                           # Number of y-volumes
    xlims = (0, 1)                    # Horizontal limits (left, right)
    ylims = (0, 1)                    # Vertical limits (bottom, top)
    stretch = (1, 1)                  # Stretch factor (sx, sy[, sz])
    grid = create_grid(T, N; Nx, Ny, xlims, ylims, stretch)

    # Discretization parameters
    order4 = false                    # Use 4th order in space (otherwise 2nd order)
    α = 81                            # Richardson extrapolation factor = 3^4
    β = 9 / 8                         # Interpolation factor
    discretization = Operators{T}(; order4, α, β)

    # Rom parameters
    use_rom = false                   # Use reduced order model
    rom_type = "POD"                  # "POD", "Fourier"
    M = 10                            # Number of ROM velocity modes
    Mp = 10                           # Number of ROM pressure modes
    precompute_convection = true      # Precomputed convection matrices
    precompute_diffusion = true       # Precomputed diffusion matrices
    precompute_force = true           # Precomputed forcing term
    t_snapshots = 0                   # Snapshots
    Δt_snapshots = false              # Gap between snapshots
    mom_cons = false                  # Momentum conserving SVD
    # ROM boundary constitions:
    # 0: homogeneous (no-slip = periodic)
    # 1: non-homogeneous = time-independent
    # 2: non-homogeneous = time-dependent
    rom_bc = 0
    weighted_norm = true              # Using finite volumes as weights
    pressure_recovery = false         # Compute pressure with PPE-ROM
    pressure_precompute = 0           # Recover pressure with FOM (0) or ROM (1)
    subtract_pressure_mean = false    # Subtract pressure mean from snapshots
    process_iteration_FOM = true      # FOM divergence, residuals, and kinetic energy
    basis_type = "default"            # "default", "svd", "direct", "snapshot"
    rom = ROM(;
        use_rom,
        rom_type,
        M,
        Mp,
        precompute_convection,
        precompute_diffusion,
        precompute_force,
        t_snapshots,
        Δt_snapshots,
        mom_cons,
        rom_bc,
        weighted_norm,
        pressure_recovery,
        pressure_precompute,
        subtract_pressure_mean,
        process_iteration_FOM,
        basis_type,
    )

    # Immersed boundary method
    use_ibm = false                    # Use immersed boundary method
    ibm = IBM(; use_ibm)

    # Time stepping
    t_start = 0                        # Start time
    t_end = 1                          # End time
    Δt = 0.01                          # Timestep
    method = RK44()                    # ODE method
    method_startup = RK44()            # Startup method for methods that are not self-starting
    nstartup = 2                       # Number of velocity fields necessary for start-up = equal to order of method
    isadaptive = false                 # Adapt timestep every n_adapt_Δt iterations
    n_adapt_Δt = 1                     # Number of iterations between timestep adjustment
    CFL = 0.5                          # CFL number for adaptive methods
    time = Time{T}(;
        t_start,
        t_end,
        Δt,
        method,
        method_startup,
        nstartup,
        isadaptive,
        n_adapt_Δt,
        CFL,
    )

    # Solver settings
    pressure_solver = DirectPressureSolver{T}()    # Pressure solver
    # pressure_solver = CGPressureSolver{T}()      # Pressure solver
    # pressure_solver = FourierPressureSolver{T}() # Pressure solver
    p_initial = true                 # Calculate compatible IC for the pressure
    p_add_solve = true               # Additional pressure solve to make it same order as velocity
    nonlinear_acc = 1e-10            # Absolute accuracy
    nonlinear_relacc = 1e-14         # Relative accuracy
    nonlinear_maxit = 10             # Maximum number of iterations
    # "no": Replace iteration matrix with I/Δt (no Jacobian)
    # "approximate": Build Jacobian once before iterations only
    # "full": Build Jacobian at each iteration
    nonlinear_Newton = "approximate"
    Jacobian_type = "newton"         # Linearization: "picard", "newton"
    nonlinear_startingvalues = false # Extrapolate values from last time step to get accurate initial guess (for unsteady problems only)
    nPicard = 2                      # Number of Picard steps before switching to Newton when linearization is Newton (for steady problems only)
    solver_settings = SolverSettings{T}(;
        pressure_solver,
        p_initial,
        p_add_solve,
        nonlinear_acc,
        nonlinear_relacc,
        nonlinear_maxit,
        nonlinear_Newton,
        Jacobian_type,
        nonlinear_startingvalues,
        nPicard,
    )

    # Boundary conditions
    bc_unsteady = false
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        k = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        e = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ν = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    )
    u_bc(x, y, t, setup) = y ≈ setup.grid.ylims[2] ? 1 : 0
    v_bc(x, y, t, setup) = 0
    bc = create_boundary_conditions(T; bc_unsteady, bc_type, u_bc, v_bc)

    # Initial conditions
    initial_velocity_u(x, y) = 0
    initial_velocity_v(x, y) = 0
    initial_pressure(x, y) = 0
    @pack! case = initial_velocity_u, initial_velocity_v, initial_pressure

    # Forcing parameters
    bodyforce_u(x, y) = 0
    bodyforce_v(x, y) = 0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

    # Iteration processors 
    logger = Logger()                        # Prints time step information
    # real_time_plotter = RealTimePlotter(; 
    #     nupdate = 5,                         # Number of iterations between real time plots
    #     fieldname = :vorticity,              # Quantity for real time plotting
    #     # fieldname = :quiver,                 # Quantity for real time plotting 
    #     # fieldname = :vorticity,              # Quantity for real time plotting 
    #     # fieldname = :pressure,               # Quantity for real time plotting 
    #     # fieldname = :streamfunction,         # Quantity for real time plotting 
    # )
    # vtk_writer = VTKWriter(;
    #     nupdate = 5,                         # Number of iterations between VTK writings 
    #     dir = "output/$name",                # Output directory
    #     filename = "solution",               # Output file name (without extension)
    # )
    # processors = [logger, real_time_plotter, vtk_writer] 
    processors = [logger] 

    # Final setup
    setup = Setup{T,N}(;
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

    for problem ∈ [SteadyStateProblem(), UnsteadyProblem()]
        setup.case.problem = problem

        ## Prepare
        build_operators!(setup);
        V₀, p₀, t₀ = create_initial_conditions(setup);

        ## Solve problem
        problem = setup.case.problem;
        V, p = solve(problem, setup, V₀, p₀);

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)
    end
end
