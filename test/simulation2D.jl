# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Simulation 2D" begin
    lid_vel = 1.0 # Lid velocity
    u_bc(x, y, t) = y ≈ 1.0 ? lid_vel : 0.0
    v_bc(x, y, t) = 0.0
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    )

    x = cosine_grid(0.0, 1.0, 25)
    y = cosine_grid(0.0, 1.0, 25)

    setup = Setup(x, y; u_bc, v_bc, bc_type)

    t_start, t_end = tlims = (0.0, 0.5)

    initial_velocity_u(x, y) = 0.0
    initial_velocity_v(x, y) = 0.0
    initial_pressure(x, y) = 0.0
    V₀, p₀ = create_initial_conditions(
        setup,
        t_start;
        initial_velocity_u,
        initial_velocity_v,
        initial_pressure,
    )

    @testset "Steady state problem" begin
        problem = SteadyStateProblem(setup, V₀, p₀)
        V, p = solve(problem)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < lid_vel
    end

    # Iteration processors
    logger = Logger()
    tracer = QuantityTracer()
    processors = [logger, tracer]

    @testset "Unsteady problem" begin
        problem = UnsteadyProblem(setup, V₀, p₀, tlims)
        V, p = solve(problem, RK44(); Δt = 0.01, processors)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < lid_vel

        # Check for steady state convergence
        @test tracer.umom[end] < 1e-10
        @test tracer.vmom[end] < 1e-10
    end
end
