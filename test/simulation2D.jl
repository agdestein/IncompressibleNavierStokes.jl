# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Simulation 2D" begin
    @info "Testing 2D simulation"

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
        initial_velocity_u,
        initial_velocity_v,
        t_start;
        initial_pressure,
    )

    @testset "Steady state problem" begin
        V, p = solve_steady_state(setup, V₀, p₀)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < lid_vel
    end

    # Iteration processors
    processors = (timelogger(),)

    @testset "Unsteady problem" begin
        V, p, outputs = solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.01, processors)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < lid_vel
    end
end
