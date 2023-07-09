# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Simulation 3D" begin
    @info "Testing 3D simulation"
    lid_vel = [1.0, 0.0, 0.2] # Lid velocity
    u_bc(x, y, z, t) = y ≈ 1.0 ? lid_vel[1] : 0.0
    v_bc(x, y, z, t) = 0.0
    w_bc(x, y, z, t) = y ≈ 1.0 ? lid_vel[3] : 0.0

    x = LinRange(0.0, 1.0, 25)
    y = LinRange(0.0, 1.0, 25)
    z = LinRange(-0.2, 0.2, 10)

    setup = Setup(x, y, z; u_bc, v_bc, w_bc)

    t_start, t_end = tlims = (0.0, 0.5)

    initial_velocity_u(x, y, z) = 0.0
    initial_velocity_v(x, y, z) = 0.0
    initial_velocity_w(x, y, z) = 0.0
    initial_pressure(x, y, z) = 0.0
    V₀, p₀ = create_initial_conditions(
        setup,
        initial_velocity_u,
        initial_velocity_v,
        initial_velocity_w,
        t_start;
        initial_pressure,
    )

    @testset "Steady state problem" begin
        V, p = solve_steady_state(setup, V₀, p₀)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < norm(lid_vel)
    end

    # Iteration processors
    processors = (step_logger(),)

    @testset "Unsteady problem" begin
        V, p, outputs = solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.01, processors)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < norm(lid_vel)
    end
end
