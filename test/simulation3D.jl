# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Simulation 3D" begin
    lid_vel = [1.0, 0.0, 0.2] # Lid velocity
    u_bc(x, y, z, t) = y ≈ 1.0 ? lid_vel[1] : 0.0
    v_bc(x, y, z, t) = 0.0
    w_bc(x, y, z, t) = y ≈ 1.0 ? lid_vel[3] : 0.0
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
    )

    x = LinRange(0.0, 1.0, 25)
    y = LinRange(0.0, 1.0, 25)
    z = LinRange(-0.2, 0.2, 10)

    setup = Setup(x, y, z; u_bc, v_bc, w_bc, bc_type)

    t_start, t_end = tlims = (0.0, 0.5)

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

    @testset "Steady state problem" begin
        problem = SteadyStateProblem(setup, V₀, p₀)
        V, p = @time solve(problem)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < norm(lid_vel)
    end

    # Iteration processors
    tracer = QuantityTracer()
    processors = [tracer]

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
