# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Models" begin
    # Floating point type for simulations
    T = Float64

    ## Grid parameters
    Nx = 25                           # Number of x-volumes
    Ny = 25                           # Number of y-volumes
    grid = create_grid(
        T,
        Nx,
        Ny;
        xlims = (0, 1),               # Horizontal limits (left, right)
        ylims = (0, 1),               # Vertical limits (bottom, top)
        stretch = (1, 1),             # Stretch factor (sx, sy[, sz])
    )

    ## Solver settings
    solver_settings = SolverSettings{T}(;
        pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
        # pressure_solver = CGPressureSolver{T}(),      # Pressure solver
        # pressure_solver = FourierPressureSolver{T}(), # Pressure solver
        p_add_solve = true,                             # Additional pressure solve for second order pressure
        abstol = 1e-10,                                 # Absolute accuracy
        reltol = 1e-14,                                 # Relative accuracy
        maxiter = 10,                                   # Maximum number of iterations
        # :no: Replace iteration matrix with I/Δt (no Jacobian)
        # :approximate: Build Jacobian once before iterations only
        # :full: Build Jacobian at each iteration
        newton_type = :approximate,
    )

    ## Boundary conditions
    lid_vel = 1.0 # Lid velocity
    u_bc(x, y, t, setup) = y ≈ setup.grid.ylims[2] ? lid_vel : 0.0
    v_bc(x, y, t, setup) = zero(x)
    bc = create_boundary_conditions(
        T,
        u_bc,
        v_bc;
        bc_unsteady = false,
        bc_type = (;
            u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            k = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            e = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            ν = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ),
    )

    ## Forcing parameters
    bodyforce_u(x, y) = 0
    bodyforce_v(x, y) = 0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

    ## Initial conditions
    initial_velocity_u(x, y) = 0
    initial_velocity_v(x, y) = 0
    initial_pressure(x, y) = 0

    ## Time interval
    t_start, t_end = tlims = (0.0, 0.5)

    ## Iteration processors
    logger = Logger()
    tracer = QuantityTracer()
    processors = [logger, tracer]

    ## Viscosity models
    Re = 1000
    lam = LaminarModel{T}(; Re)
    kϵ = KEpsilonModel{T}(; Re)
    ml = MixingLengthModel{T}(; Re)
    smag = SmagorinskyModel{T}(; Re)
    qr = QRModel{T}(; Re)

    ## Convection models
    noreg = NoRegConvectionModel{T}()
    c2 = C2ConvectionModel{T}()
    c4 = C4ConvectionModel{T}()
    leray = LerayConvectionModel{T}()

    models = [
        (lam, noreg)
        (kϵ, noreg)
        (ml, noreg)
        (smag, noreg)
        (qr, noreg)
        (lam, c2)
        (lam, c4)
        (lam, leray)
    ]

    for (viscosity_model, convection_model) in models
        @testset "$(typeof(viscosity_model)) $(typeof(convection_model))" begin
            setup = Setup{T,2}(;
                viscosity_model,
                convection_model,
                grid,
                force,
                solver_settings,
                bc,
            )

            build_operators!(setup)

            V₀, p₀ = create_initial_conditions(
                setup,
                t_start;
                initial_velocity_u,
                initial_velocity_v,
                initial_pressure,
            )

            problem = SteadyStateProblem(setup, V₀, p₀)
            @test V, p = solve(problem; processors)

            # Check that the average velocity is smaller than the lid velocity
            @test sum(abs, V) / length(V) < lid_vel

            problem = UnsteadyProblem(setup, V₀, p₀, tlims)
            V, p = solve(problem, RK44(); Δt = 0.01, processors)

            # Check that the average velocity is smaller than the lid velocity
            sum(abs, V) / length(V) < lid_vel

            # Check for steady state convergence
            @test tracer.umom[end] < 1e-10
            @test tracer.vmom[end] < 1e-10
        end
    end
end
