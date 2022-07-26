# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Simulation 3D" begin
    # Floating point type for simulations
    T = Float64

    ## Viscosity model
    viscosity_model = LaminarModel{T}(; Re = 1000)
    # viscosity_model = MixingLengthModel{T}(; Re = 1000)
    # viscosity_model = SmagorinskyModel{T}(; Re = 1000)
    # viscosity_model = QRModel{T}(; Re = 1000)

    ## Grid parameters
    x = stretched_grid(0.0, 1.0, 25)
    y = stretched_grid(0.0, 1.0, 25)
    z = stretched_grid(-0.2, 0.2, 10)
    grid = create_grid(x, y, z; T)

    ## Boundary conditions
    lid_vel = [1.0, 0.0, 0.2] # Lid velocity
    u_bc(x, y, z, t) = y ≈ grid.ylims[2] ? lid_vel[1] : 0.0
    v_bc(x, y, z, t) = 0.0
    w_bc(x, y, z, t) = y ≈ grid.ylims[2] ? lid_vel[3] : 0.0
    bc = create_boundary_conditions(
        u_bc,
        v_bc,
        w_bc;
        bc_unsteady = false,
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
        ),
        T,
    )

    ## Forcing parameters
    bodyforce_u(x, y, z) = 0.0
    bodyforce_v(x, y, z) = 0.0
    bodyforce_w(x, y, z) = 0.0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v, bodyforce_w)

    ## Pressure solver
    pressure_solver = DirectPressureSolver{T}()
    # pressure_solver = CGPressureSolver{T}()
    # pressure_solver = FourierPressureSolver{T}()

    ## Build setup and assemble operators
    setup =
        Setup{T,3}(; viscosity_model,  grid, force, pressure_solver, bc)
    build_operators!(setup)

    ## Time interval
    t_start, t_end = tlims = (0.0, 0.5)

    ## Initial conditions
    initial_velocity_u(x, y, z) = 0.0
    initial_velocity_v(x, y, z) = 0.0
    initial_velocity_w(x, y, z) = 0.0
    initial_pressure(x, y, z) = 0.0
    V₀, p₀ = create_initial_conditions(
        setup,
        t_start;
        initial_velocity_u,
        initial_velocity_v,
        initial_velocity_w,
        initial_pressure,
    )

    ## Iteration processors
    logger = Logger()
    tracer = QuantityTracer()
    processors = [logger, tracer]

    @testset "Unsteady problem" begin
        problem = UnsteadyProblem(setup, V₀, p₀, tlims)
        V, p = @time solve(problem, RK44(); Δt = 0.01, processors)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < norm(lid_vel)

        # Check for steady state convergence
        @test tracer.umom[end] < 1e-10
        @test tracer.vmom[end] < 1e-10
        @test tracer.wmom[end] < 1e-10
    end
end
