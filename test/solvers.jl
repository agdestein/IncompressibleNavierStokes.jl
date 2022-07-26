@testset "Taylor-Green vortex" begin
    T = Float64

    Re = 500
    viscosity_model = LaminarModel{T}(; Re)

    ## Grid
    x = stretched_grid(0, 2π, 50)
    y = stretched_grid(0, 2π, 50)
    grid = create_grid(x, y; T)

    ## Boundary conditions
    u_bc(x, y, t) = zero(x)
    v_bc(x, y, t) = zero(x)
    bc = create_boundary_conditions(
        u_bc,
        v_bc;
        bc_unsteady = false,
        bc_type = (;
            u = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
            v = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        ),
        T,
    )

    ## Forcing parameters
    bodyforce_u(x, y) = 0
    bodyforce_v(x, y) = 0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

    ## Pressure solver
    pressure_solver = FourierPressureSolver{T}()

    ## Build setup and assemble operators
    setup =
        Setup{T,2}(; viscosity_model,  grid, force, pressure_solver, bc)
    build_operators!(setup)

    ## Time interval
    t_start, t_end = tlims = (0.0, 5.0)

    ## Initial conditions
    initial_velocity_u(x, y) = cos(x)sin(y)
    initial_velocity_v(x, y) = -sin(x)cos(y)
    initial_pressure(x, y) = -1 / 4 * (cos(2x) + cos(2y))
    V₀, p₀ = create_initial_conditions(
        setup,
        t_start;
        initial_velocity_u,
        initial_velocity_v,
        initial_pressure,
    )

    # Exact solutions
    F(t) = exp(-2t / Re)
    u(x, y, t) = initial_velocity_u(x, y) * F(t)
    v(x, y, t) = initial_velocity_v(x, y) * F(t)
    (; xu, yu, xv, yv) = grid
    uₕ = u.(xu, yu, t_end)
    vₕ = v.(xv, yv, t_end)
    V_exact = [uₕ[:]; vₕ[:]]

    @testset "Unsteady solvers" begin
        problem = UnsteadyProblem(setup, V₀, p₀, tlims)

        @testset "Explicit Runge Kutta" begin
            V, p = solve(problem, RK44(); Δt = 0.01)
            @test norm(V - V_exact) / norm(V_exact) < 1e-4
        end

        @testset "Implicit Runge Kutta" begin
            V, p = solve(problem, RIA2(); Δt = 0.01)
            @test_broken norm(V - V_exact) / norm(V_exact) < 1e-3
        end

        @testset "One-leg beta method" begin
            V, p = solve(problem, OneLegMethod{T}(); method_startup = RK44(), Δt = 0.01)
            @test norm(V - V_exact) / norm(V_exact) < 1e-4
        end

        @testset "Adams-Bashforth Crank-Nicolson" begin
            V, p = solve(
                problem,
                AdamsBashforthCrankNicolsonMethod{T}();
                method_startup = RK44(),
                Δt = 0.01,
            )
            @test norm(V - V_exact) / norm(V_exact) < 1e-4
        end
    end
end
