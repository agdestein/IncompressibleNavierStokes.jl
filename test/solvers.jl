@testset "Solvers" begin
    T = Float64
    Re = 500.0

    n = 50
    x = LinRange(0, 2π, n + 1)
    y = LinRange(0, 2π, n + 1)
    setup = Setup(x, y; Re)

    psolver = SpectralPressureSolver(setup)

    t_start, t_end = tlims = (0.0, 5.0)

    initial_velocity_u(x, y) = cos(x)sin(y)
    initial_velocity_v(x, y) = -sin(x)cos(y)
    initial_pressure(x, y) = -1 / 4 * (cos(2x) + cos(2y))
    V₀, p₀ = create_initial_conditions(
        setup,
        initial_velocity_u,
        initial_velocity_v,
        t_start;
        initial_pressure,
        psolver,
    )

    @testset "Steady state" begin
        @info "Testing steady state solver"
        V, p = solve_steady_state(setup, V₀, p₀)
        uₕ = V[setup.grid.indu]
        vₕ = V[setup.grid.indv]
        @test norm(uₕ .- mean(uₕ)) / mean(uₕ) < 1e-8
        @test norm(vₕ .- mean(vₕ)) / mean(vₕ) < 1e-8
    end

    # Exact solutions
    F(t) = exp(-2t / Re)
    u(x, y, t) = initial_velocity_u(x, y) * F(t)
    v(x, y, t) = initial_velocity_v(x, y) * F(t)
    (; xu, yu, xv, yv) = setup.grid
    uₕ = u.(xu, yu, t_end)
    vₕ = v.(xv, yv, t_end)
    V_exact = [uₕ[:]; vₕ[:]]

    @testset "Unsteady solvers" begin
        @testset "Explicit Runge Kutta" begin
            @info "Testing explicit Runge-Kutta, out-of-place version"
            state, outputs =
                solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.01, psolver, inplace = false)
            @test norm(state.u - u_exact) / norm(u_exact) < 1e-4
            @info "Testing explicit Runge-Kutta, in-place version"
            stateip, outputsip =
                solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.01, psolver, inplace = true)
            @test stateip.u ≈ state.u
            @test stateip.p ≈ state.p
        end

        @testset "Implicit Runge Kutta" begin
            @info "Testing implicit Runge-Kutta, out-of-place version"
            @test_broken solve_unsteady(
                setup,
                V₀,
                p₀,
                tlims;
                method = RIA2(),
                Δt = 0.01,
                psolver,
                inplace = false,
            ) isa Tuple
            @info "Testing implicit Runge-Kutta, in-place version"
            (; u, p, t), outputs = solve_unsteady(
                setup,
                V₀,
                p₀,
                tlims;
                method = RIA2(),
                Δt = 0.01,
                psolver,
                inplace = true,
                processors = (timelogger(),),
            )
            @test_broken norm(u - u_exact) / norm(u_exact) < 1e-3
        end

        @testset "One-leg beta method" begin
            @info "Testing one-leg beta method, out-of-place version"
            state, outputs = solve_unsteady(
                setup,
                V₀,
                p₀,
                tlims;
                method = OneLegMethod(T),
                Δt = 0.01,
                psolver,
                inplace = false,
            )
            @test norm(state.u - u_exact) / norm(u_exact) < 1e-4
            @info "Testing one-leg beta method, in-place version"
            stateip, outputsip = solve_unsteady(
                setup,
                V₀,
                p₀,
                tlims;
                method = OneLegMethod(T),
                Δt = 0.01,
                psolver,
                inplace = true,
            )
            @test stateip.u ≈ state.u
            @test stateip.p ≈ state.p
        end

        @testset "Adams-Bashforth Crank-Nicolson" begin
            @info "Testing Adams-Bashforth Crank-Nicolson method, out-of-place version"
            state, outputs = solve_unsteady(
                setup,
                V₀,
                p₀,
                tlims;
                method = AdamsBashforthCrankNicolsonMethod(T),
                Δt = 0.01,
                psolver,
                inplace = false,
            )
            @test norm(state.u - u_exact) / norm(u_exact) < 1e-4
            @info "Testing Adams-Bashforth Crank-Nicolson method, in-place version"
            stateip, outputs = solve_unsteady(
                setup,
                V₀,
                p₀,
                tlims;
                method = AdamsBashforthCrankNicolsonMethod(T),
                Δt = 0.01,
                psolver,
                inplace = true,
            )
            @test stateip.u ≈ state.u
            @test stateip.p ≈ state.p
        end
    end
end
