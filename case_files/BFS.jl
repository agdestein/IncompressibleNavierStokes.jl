"""
    setup = BFS()

Setup for unsteady Backward Facing Step case (BFS).
"""
function BFS()
    # Floating point type for simulations
    T = Float64

    # Case information
    name = "BFS"
    problem = UnsteadyProblem()
    # problem = SteadyStateProblem()
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

    # Grid parameters
    Nx = 400                           # Number of volumes in the x-direction
    Ny = 40                            # Number of volumes in the y-direction
    x1 = 0                             # Left
    x2 = 10                            # Right
    y1 = -0.5                          # Bottom
    y2 = 0.5                           # Top
    sx = 1                             # Stretch factor in x-direction
    sy = 1                             # Stretch factor in y-direction
    grid = Grid{T}(; Nx, Ny, x1, x2, y1, y2, sx, sy)

    # Discretization parameters
    order4 = false           # Use 4th order in space (otherwise 2nd order)
    α = 81                   # Richardson extrapolation factor = 3^4
    β = 9 / 8                # Interpolation factor
    discretization = Discretization{T}(; order4, α, β)

    # Forcing parameters
    x_c = 0                           # X-coordinate of body
    y_c = 0                           # Y-coordinate of body
    Ct = 0                            # Thrust coefficient for actuator disk computations
    D = 1                             # Actuator disk diameter
    isforce = false                   # Presence of a body force
    force_unsteady = false            # Steady (0) or unsteady (1) force
    force = Force{T}(; x_c, y_c, Ct, D, isforce, force_unsteady)

    # Rom parameters
    use_rom = false                     # Use reduced order model
    rom_type = "POD"                    # "POD", "Fourier"
    M = 10                              # Number of velocity modes for reduced order model
    Mp = 10                             # Number of pressure modes for reduced order model
    precompute_convection = true        # Precomputed convection matrices
    precompute_diffusion = true         # Precomputed diffusion matrices
    precompute_force = true             # Precomputed forcing term
    t_snapshots = 0                     # Snapshots
    Δt_snapshots = false
    mom_cons = false                    # Momentum conserving SVD
    rom_bc = 0                          # 0: homogeneous (no-slip = periodic) 1: non-homogeneous = time-independent 2: non-homogeneous = time-dependent
    weighted_norm = true                # Use weighted norm (using finite volumes as weights)
    pressure_recovery = false           # False: no pressure computation, true: compute pressure with PPE-ROM
    pressure_precompute = 0             # In case of pressure_recovery: compute RHS Poisson equation based on FOM (0) or ROM (1)
    subtract_pressure_mean = false      # Subtract pressure mean from snapshots
    process_iteration_FOM = true        # Compute divergence = residuals = kinetic energy etc. on FOM level
    basis_type = "default"              # "default" (code chooses), "svd", "direct", "snapshot"
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
    t_end = 4                          # End time
    Δt = 0.02                          # Timestep
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

    # Pressure solver: DirectPressureSolver(), CGPressureSolver(), FourierPressureSolver()
    pressure_solver = DirectPressureSolver()
    p_initial = true                 # Calculate compatible IC for the pressure
    p_add_solve = true               # Additional pressure solve to make it same order as velocity
    # Accuracy for non-linear solves (method 62 = 72 = 9)
    nonlinear_acc = 1e-10            # Absolute accuracy
    nonlinear_relacc = 1e-14         # Relative accuracy
    nonlinear_maxit = 10             # Maximum number of iterations
    # "no": Replace iteration matrix with I/Δt (no Jacobian)
    # "approximate": Build Jacobian once before iterations only
    # "full": Build Jacobian at each iteration
    nonlinear_Newton = "full"
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

    # Visualization settings
    plotgrid = false                   # Plot gridlines and pressure points
    do_rtp = true                      # Real time plotting
    rtp_type = "vorticity"             # "velocity", "quiver", "vorticity", "pressure", or "streamfunction"
    rtp_n = 10                         # Number of iterations between real time plots
    visualization = Visualization(; plotgrid, do_rtp, rtp_type, rtp_n)

    function initialize_processor(stepper)
        @unpack V, p, t, setup = stepper
        if setup.visualization.do_rtp
            rtp = initialize_rtp(setup, V, p, t)
        else
            rtp = nothing
        end
        # Estimate number of time steps that will be taken
        nt = ceil(Int, (t_end - t_start) / Δt)

        momentum!(F, nothing, V, V, p, t, setup, momentum_cache)
        maxres = maximum(abs.(F))

        
        println("n), t = $t, maxres = $maxres")
        # println("t = $t")

        (; rtp, nt)
    end

    function process!(processor, stepper)

        @unpack do_rtp, rtp_n = setup.visualization
        
        # Calculate mass, momentum and energy
        # maxdiv, umom, vmom, k = compute_conservation(V, t, setup)

        # Residual (in Finite Volume form)
        # For k-ϵ model residual also contains k and ϵ terms
        if !isa(model, KEpsilonModel)
            # Norm of residual
            momentum!(F, nothing, V, V, p, t, setup, momentum_cache)
            maxres = maximum(abs.(F))
        end
        
        println("n = $(stepper.n), t = $t, maxres = $maxres")
        # println("t = $t")

        if do_rtp && mod(n, rtp_n) == 0
            @unpack setup, V, p, t = stepper
            update_rtp!(rtp, setup, V, p, t)
        end
    end

    """
        bc_type()

    left/right: x-direction
    low/up: y-direction
    """
    function bc_type()
        bc_unsteady = false

        u = (; left = :dirichlet, right = :pressure, low = :dirichlet, up = :dirichlet)
        v = (; left = :dirichlet, right = :symmetric, low = :dirichlet, up = :dirichlet)

        # u = (; left = :periodic, right = :periodic, low = :periodic, up = :periodic)
        # v = (; left = :periodic, right = :periodic, low = :periodic, up = :periodic)

        k = (; left = :dirichlet, right = :dirichlet, low = :dirichlet, up = :dirichlet)
        e = (; left = :dirichlet, right = :dirichlet, low = :dirichlet, up = :dirichlet)

        # Values set below can be either Dirichlet or Neumann value,
        # Depending on B.C. set above. in case of Neumann (symmetry, pressure)
        # One uses normally zero gradient
        # Neumann B.C. used to extrapolate values to the boundary
        # Change only in case of periodic to :periodic, otherwise leave at :symmetric
        ν = (;
            left = :symmetric,
            right = :symmetric,
            low = :symmetric,
            up = :symmetric,
            back = :symmetric,
            front = :symmetric,
        )

        (; bc_unsteady, u, v, k, e, ν)
    end

    """
        u = u_bc(x, y, t, setup[, tol])

    Compute boundary conditions for `u` at point `(x, y)` at time `t`.
    """
    function u_bc(x, y, t, setup, tol = 1e-10)
        if ≈(x, setup.grid.x1; rtol = tol) && y ≥ 0
            24y * (1 // 2 - y)
        else
            zero(y)
        end
    end

    """
        v = v_bc(x, y, t, setup)

    Compute boundary conditions for `u` at point `(x, y)` at time `t`.
    """
    function v_bc(x, y, t, setup)
        v = 0
    end

    bc = BC{T}(; bc_type, u_bc, v_bc)

    """
        u = initial_velocity_u(x, y, setup)

    Get initial velocity `(u, v)` at point `(x, y)`.
    """
    function initial_velocity_u(x, y, setup, tol = 1e-10)
        # Initial velocity field BFS (extend inflow)
        y ≥ 0 ? 24y * (1 // 2 - y) : zero(y)
    end

    """
    v = initial_velocity_v(x, y, setup)

    Get initial velocity `v` at point `(x, y)`.
    """
    function initial_velocity_v(x, y, setup)
        # Initial velocity field BFS (constant)
        v = 0
    end

    """
    p = initial_pressure(x, y, setup)

    Get initial pressure `p` at point `(x, y)`. Should in principle NOT be prescribed. Will be calculated if `p_initial`.
    """
    function initial_pressure(x, y, setup)
        p = 0
    end

    case.initial_velocity_u = initial_velocity_u
    case.initial_velocity_v = initial_velocity_v
    case.initial_pressure = initial_pressure

    """
        Fx, dFx = bodyforce_x(V, t, setup, getJacobian = false)

    Get body force (`x`-component) at point `(x, y)` at time `t`.
    """
    function bodyforce_x(x, y, t, setup, getJacobian = false)
        Fx = 0
        dFx = 0
        Fx, dFx
    end

    """
    Fy, dFy = bodyforce_y(x, y, t, setup, getJacobian = false)

    Get body force (`y`-component) at point `(x, y)` at time `t`.
    """
    function bodyforce_y(V, t, setup, getJacobian = false) end

    function Fp(x, y, t, setup, getJacobian = false)
        # At pressure points, for pressure solution
    end

    force.bodyforce_x = bodyforce_x
    force.bodyforce_y = bodyforce_y

    """
        x, y = mesh(setup)

    Build mesh points `x` and `y`.
    """
    function create_mesh(setup)
        # Uniform mesh size x-direction
        @unpack Nx, sx, x1, x2 = setup.grid
        L_x = x2 - x1
        deltax = L_x / Nx

        # Uniform mesh size y-direction
        @unpack Ny, sy, y1, y2 = setup.grid
        L_y = y2 - y1
        deltay = L_y / Ny

        x, _ = nonuniform_grid(deltax, x1, x2, sx)
        y, _ = nonuniform_grid(deltay, y1, y2, sy)

        x, y
    end

    grid.create_mesh = create_mesh

    Setup{T}(;
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
        visualization,
        bc,
    )
end
