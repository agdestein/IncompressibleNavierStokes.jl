"""
    setup = LDC()

Setup for Lid-Driven Cavity case (LDC).
"""
function LDC()
    # Construct Setup object, containing substructures with default values
    setup = Setup{Float64}()

    # Case information
    setup.case.name = "LDC"
    setup.case.problem = SteadyStateProblem()
    setup.case.regularization = "no"

    # Physical properties
    setup.fluid.Re = 1000                          # Reynolds number
    setup.fluid.U1 = 1                             # Velocity scales
    setup.fluid.U2 = 1                             # Velocity scales
    setup.fluid.d_layer = 1                        # Thickness of layer

    # Viscosity model
    setup.model = LaminarModel()

    # Grid parameters
    setup.grid.Nx = 80                             # Number of volumes in the x-direction
    setup.grid.Ny = 80                             # Number of volumes in the y-direction
    setup.grid.x1 = 0                              # Left
    setup.grid.x2 = 1                              # Right
    setup.grid.y1 = 0                              # Bottom
    setup.grid.y2 = 1                              # Top
    setup.grid.sx = 1                              # Stretch factor in x-direction
    setup.grid.sy = 1                              # Stretch factor in y-direction

    # Discretization parameters
    setup.discretization.order4 = false            # Use 4th order in space (otherwise 2nd order)
    setup.discretization.α = 81                    # Richardson extrapolation factor = 3^4
    setup.discretization.β = 9 / 8                 # Interpolation factor

    # Forcing parameters
    setup.force.x_c = 0                            # X-coordinate of body
    setup.force.y_c = 0                            # Y-coordinate of body
    setup.force.Ct = 0                             # Thrust coefficient for actuator disk computations
    setup.force.D = 1                              # Actuator disk diameter
    setup.force.isforce = false                    # Presence of a body force
    setup.force.force_unsteady = false             # Steady (false) or unsteady (true) force

    # Rom parameters
    setup.rom.use_rom = false                      # Use reduced order model
    setup.rom.rom_type = "POD"                     # "POD",  "Fourier"
    setup.rom.M = 10                               # Number of velocity modes for reduced order model
    setup.rom.Mp = 10                              # Number of pressure modes for reduced order model
    setup.rom.precompute_convection = true         # Precomputed convection matrices
    setup.rom.precompute_diffusion = true          # Precomputed diffusion matrices
    setup.rom.precompute_force = true              # Precomputed forcing term
    setup.rom.t_snapshots = 0                      # Snapshots
    setup.rom.Δt_snapshots = false
    setup.rom.mom_cons = false                     # Momentum conserving SVD
    setup.rom.rom_bc = 0                           # 0: homogeneous (no-slip = periodic) 1: non-homogeneous = time-independent 2: non-homogeneous = time-dependent
    setup.rom.weighted_norm = true                 # Use weighted norm (using finite volumes as weights)
    setup.rom.pressure_recovery = false            # False: no pressure computation, true: compute pressure with PPE-ROM
    setup.rom.pressure_precompute = 0              # In case of pressure_recovery: compute RHS Poisson equation based on FOM (0) or ROM (1)
    setup.rom.subtract_pressure_mean = false       # Subtract pressure mean from snapshots
    setup.rom.process_iteration_FOM = true         # Compute divergence = residuals = kinetic energy etc. on FOM level
    setup.rom.basis_type = "default"               # "default" (code chooses), "svd", "direct", "snapshot"

    # Immersed boundary method
    setup.ibm.ibm = false                          # Use immersed boundary method

    # Time marching
    setup.time.t_start = 0                         # Start time
    setup.time.t_end = 10                          # End time
    setup.time.Δt = 0.01                           # Timestep
    setup.time.time_stepper = RK44()               # Time stepper
    setup.time.time_stepper_startup = RK44()       # Startup method for methods that are not self-starting
    setup.time.nstartup = 2                        # Number of velocity fields necessary for start-up = equal to order of method
    setup.time.isadaptive = false                  # Adapt timestep every n_adapt_Δt iterations
    setup.time.n_adapt_Δt = 1                      # Number of iterations between timestep adjustment
    setup.time.CFL = 0.5                           # CFL number for adaptive methods

    # Solver settings
    setup.solver_settings.pressure_solver = DirectPressureSolver() # Pressure solver
    setup.solver_settings.p_initial = true         # Calculate compatible IC for the pressure
    setup.solver_settings.p_add_solve = true       # Additional pressure solve to make it same order as velocity

    # Accuracy for non-linear solves (method 62 = 72 = 9)
    setup.solver_settings.nonlinear_acc = 1e-14    # Absolute accuracy
    setup.solver_settings.nonlinear_relacc = 1e-14 # Relative accuracy
    setup.solver_settings.nonlinear_maxit = 10     # Maximum number of iterations

    # "no": do not compute Jacobian, but approximate iteration matrix with I/Δt
    # "approximate: approximate Newton build Jacobian once at beginning of nonlinear iterations
    # "full": full Newton build Jacobian at each iteration
    setup.solver_settings.nonlinear_Newton = "approximate"

    setup.solver_settings.Jacobian_type = "newton"           # "picard": Picard linearization, "newton": Newton linearization
    setup.solver_settings.nonlinear_startingvalues = false   # Extrapolate values from last time step to get accurate initial guess (for unsteady problems only)
    setup.solver_settings.nPicard = 2                        # Number of Picard steps before switching to Newton when linearization is Newton (for steady problems only)

    # Output files
    setup.output.save_results = false                        # Save results
    setup.output.savepath = "results"                        # Path for saving
    setup.output.save_unsteady = false                       # Save intermediate time steps

    # Visualization settings
    setup.visualization.plotgrid = false                     # Plot gridlines and pressure points
    setup.visualization.do_rtp = false                       # Real time plotting
    setup.visualization.rtp_type = "velocity"                # "velocity", "quiver", "vorticity", or "pressure"
    setup.visualization.rtp_n = 1                            # Number of iterations between real time plots


    """
        bc_type()

    left/right: x-direction
    low/up: y-direction
    """
    setup.bc.bc_type = function bc_type()
        bc_unsteady = false

        u = (; left = :dirichlet, right = :dirichlet, low = :dirichlet, up = :dirichlet)
        v = (; left = :dirichlet, right = :dirichlet, low = :dirichlet, up = :dirichlet)
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
    setup.bc.u_bc = function u_bc(x, y, t, setup, tol = 1e-10)
        u = ≈(y, setup.grid.y2; rtol = tol) ? 1 : 0
    end

    """
        v = v_bc(x, y, t, setup)

    Compute boundary conditions for `u` at point `(x, y)` at time `t`.
    """
    setup.bc.v_bc = function v_bc(x, y, t, setup)
        v = 0
    end

    """
        u = initial_velocity_u(x, y, setup)

    Get initial velocity `(u, v)` at point `(x, y)`.
    """
    setup.case.initial_velocity_u = function initial_velocity_u(x, y, setup)
        # Initial velocity field LDC (constant)
        u = 0
    end

    """
    v = initial_velocity_v(x, y, setup)

    Get initial velocity `v` at point `(x, y)`.
    """
    setup.case.initial_velocity_v = function initial_velocity_v(x, y, setup)
        # Initial velocity field LDC (constant)
        v = 0
    end

    """
    p = initial_pressure(x, y, setup)

    Get initial pressure `p` at point `(x, y)`. Should in principle NOT be prescribed. Will be calculated if `p_initial`.
    """
    setup.case.initial_pressure = function initial_pressure(x, y, setup)
        p = 0
    end

    """
        Fx, dFx = bodyforce_x(V, t, setup, getJacobian = false)

    Get body force (`x`-component) at point `(x, y)` at time `t`.
    """
    setup.force.bodyforce_x = function bodyforce_x(x, y, t, setup, getJacobian = false)
        Fx = 0
        dFx = 0
        Fx, dFx
    end

    """
    Fy, dFy = bodyforce_y(x, y, t, setup, getJacobian = false)

    Get body force (`y`-component) at point `(x, y)` at time `t`.
    """
    setup.force.bodyforce_y = function bodyforce_y(V, t, setup, getJacobian = false)
        getJacobian && error("Jacobian not available")

        # Point force
        Shih_f = x^4 - 2x^3 + x^2
        Shih_f1 = 4x^3 - 6x^2 + 2x
        Shih_f2 = 12x^2 - 12x + 2
        Shih_f3 = 24x - 12
        Shih_g = y^4 - y^2
        Shih_g1 = 4y^3 - 2y
        Shih_g2 = 12y^2 - 2

        Shih_F = 0.2x^5 - 0.5x^4 + 1 / 3 * x^3
        Shih_F1 = -4x^6 + 12x^5 - 14x^4 + 8x^3 - 2x^2
        Shih_F2 = 0.5 * (Shih_f^2)
        Shih_G1 = -24y^5 + 8y^3 - 4y

        Fy =
            -8 / Re * (24 * Shih_F + 2 * Shih_f1 * Shih_g2 + Shih_f3 * Shih_g) -
            64 * (Shih_F2 * Shih_G1 - Shih_g * Shih_g1 * Shih_F1)

        Fy, dFy
    end

    function Fp(x, y, t, setup, getJacobian = false)
        # At pressure points, for pressure solution
        Shih_p_f = x^4 - 2x^3 + x^2
        Shih_p_f1 = 4x^3 - 6x^2 + 2x
        Shih_p_f2 = 12x^2 - 12x + 2
        Shih_p_f3 = 24x - 12
        Shih_p_g = y^4 - y^2
        Shih_p_g1 = 4y^3 - 2y
        Shih_p_g2 = 12y^2 - 2
        Shih_p_g3 = 24y

        Shih_p_F = 0.2x^5 - 0.5x^4 + 1 / 3 * x^3
        Shih_p_F1 = -4x^6 + 12x^5 - 14x^4 + 8x^3 - 2x^2
        Shih_p_F2 = 0.5 * (Shih_p_f^2)
        Shih_p_G1 = -24y^5 + 8y^3 - 4y
    end

    """
        x, y = mesh(setup)

    Build mesh points `x` and `y`.
    """
    setup.grid.create_mesh = function create_mesh(setup)
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

        # Transform uniform grid to non-uniform cosine grid
        @. x = L_x / 2 * (1 - cos(π * x / L_x))
        @. y = L_y / 2 * (1 - cos(π * y / L_y))

        x, y
    end

    setup
end
