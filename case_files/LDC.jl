"""
    LDC()

Create setup for Lid-Driven Cavity case (LDC).
"""
function LDC()
    # Floating point type for simulations
    T = Float64

    # Spatial dimension
    N = 3

    # Case information
    case = Case(;
        name = "LDC",
        # problem = SteadyStateProblem(),
        problem = UnsteadyProblem(),
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
        Nx = 25,                      # Number of x-volumes
        Ny = 25,                      # Number of y-volumes
        Nz = 10,                      # Number of z-volumes
        xlims = (0, 1),               # Horizontal limits (left, right)
        ylims = (0, 1),               # Vertical limits (bottom, top)
        zlims = (-0.2, 0.2),          # Depth limits (back, front)
        stretch = (1, 1, 1),          # Stretch factor (sx, sy[, sz])
    )

    # Time stepping
    time = Time{T}(;
        t_start = 0,                           # Start time
        t_end = 10.0,                          # End time
        Δt = 0.02,                             # Timestep
        method = RK44(),                       # ODE method
        # method = RIA2(),                       # ODE method
        # method = AdamsBashforthCrankNicolsonMethod(), # ODE method
        # method = OneLegMethod(),               # ODE method
        method_startup = RK44(),               # Startup method for methods that are not self-starting
        nstartup = 2,                          # Number of necessary Vₙ₋ᵢ (= method order)
        isadaptive = false,                    # Adapt timestep every n_adapt_Δt iterations
        n_adapt_Δt = 1,                        # Number of iterations between timestep adjustment
        CFL = 0.5,                             # CFL number for adaptive methods
    )

    # Solver settings
    solver_settings = SolverSettings{T}(;
        pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
        # pressure_solver = CGPressureSolver{T}(),      # Pressure solver
        # pressure_solver = FourierPressureSolver{T}(), # Pressure solver
        p_initial = true,                               # Calculate compatible IC for the pressure
        p_add_solve = true,                             # Additional pressure solve for second order pressure
        nonlinear_acc = 1e-10,                          # Absolute accuracy
        nonlinear_relacc = 1e-14,                       # Relative accuracy
        nonlinear_maxit = 10,                           # Maximum number of iterations
        # "no": Replace iteration matrix with I/Δt (no Jacobian)
        # "approximate": Build Jacobian once before iterations only
        # "full": Build Jacobian at each iteration
        nonlinear_Newton = "approximate",
        Jacobian_type = "newton",                       # Linearization: "picard", "newton"
        nonlinear_startingvalues = false,               # Extrapolate values from last time step to get accurate initial guess (for unsteady problems only)
        nPicard = 2,                                    # Number of Picard steps before switching to Newton when linearization is Newton (for steady problems only)
    )

    # Boundary conditions
    bc_unsteady = false
    bc_type = (;
        u = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:dirichlet, :dirichlet),
        ),
        v = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:dirichlet, :dirichlet),
        ),
        w = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:dirichlet, :dirichlet),
        ),
        k = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:dirichlet, :dirichlet),
        ),
        e = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:dirichlet, :dirichlet),
        ),
        ν = (;
            x = (:dirichlet, :dirichlet),
            y = (:dirichlet, :dirichlet),
            z = (:dirichlet, :dirichlet),
        ),
    )
    u_bc(x, y, z, t, setup) = y ≈ setup.grid.ylims[2] ? 1.0 : 0.0
    v_bc(x, y, z, t, setup) = zero(x)
    w_bc(x, y, z, t, setup) = y ≈ setup.grid.ylims[2] ? 0.2 : 0.0
    bc = create_boundary_conditions(T; bc_unsteady, bc_type, u_bc, v_bc, w_bc)

    # Initial conditions
    initial_velocity_u(x, y, z) = 0
    initial_velocity_v(x, y, z) = 0
    initial_velocity_w(x, y, z) = 0
    initial_pressure(x, y, z) = 0
    @pack! case =
        initial_velocity_u, initial_velocity_v, initial_velocity_w, initial_pressure

    # Forcing parameters
    bodyforce_u(x, y, z) = 0
    bodyforce_v(x, y, z) = 0
    bodyforce_w(x, y, z) = 0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v, bodyforce_w)

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
        nupdate = 5,                         # Number of iterations between VTK writings
        dir = "output/$(case.name)",         # Output directory
        filename = "solution",               # Output file name (without extension))
    )
    tracer = QuantityTracer(; nupdate = 1)
    # processors = [logger, real_time_plotter, vtk_writer, tracer]
    processors = [logger, vtk_writer, tracer]

    # Final setup
    Setup{T,N}(;
        case,
        model,
        grid,
        force,
        time,
        solver_settings,
        processors,
        bc,
    )
end
