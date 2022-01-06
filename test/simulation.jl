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

    # Viscosity model
    model = LaminarModel{T}(; Re = 1000)
    # model = KEpsilonModel{T}(; Re = 1000)
    # model = MixingLengthModel{T}(; Re = 1000)
    # model = SmagorinskyModel{T}(; Re = 1000)
    # model = QRModel{T}(; Re = 1000)

    # Grid parameters
    Nx = 80                           # Number of x-volumes
    Ny = 80                           # Number of y-volumes
    xlims = (0, 1)                    # Horizontal limits (left, right)
    ylims = (0, 1)                    # Vertical limits (bottom, top)
    stretch = (1, 1)                  # Stretch factor (sx, sy[, sz])
    grid = create_grid(T, N; Nx, Ny, xlims, ylims, stretch)

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
        operators,
        force,
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
