# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Models" begin
    # Floating point type for simulations
    T = Float64

    ## Grid parameters
    x = stretched_grid(0.0, 1.0, 25)
    y = stretched_grid(0.0, 1.0, 25)
    grid = create_grid(x, y; T)

    ## Boundary conditions
    lid_vel = 1.0 # Lid velocity
    u_bc(x, y, t) = y ≈ grid.ylims[2] ? lid_vel : 0.0
    v_bc(x, y, t) = 0.0
    bc = create_boundary_conditions(
        u_bc,
        v_bc;
        bc_unsteady = false,
        bc_type = (;
            u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            ν = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            k = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            e = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ),
        T,
    )

    ## Forcing parameters
    bodyforce_u(x, y) = 0.0
    bodyforce_v(x, y) = 0.0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

    ## Pressure solver
    pressure_solver = DirectPressureSolver{T}()
    # pressure_solver = CGPressureSolver{T}()
    # pressure_solver = FourierPressureSolver{T}()

    ## Initial conditions
    initial_velocity_u(x, y) = 0.0
    initial_velocity_v(x, y) = 0.0
    initial_pressure(x, y) = 0.0

    ## Time interval
    t_start, t_end = tlims = (0.0, 0.5)

    ## Iteration processors
    logger = Logger()
    tracer = QuantityTracer()
    processors = [logger, tracer]

    ## Viscosity models
    Re = 1000
    lam = LaminarModel{T}(; Re)
    ml = MixingLengthModel{T}(; Re)
    smag = SmagorinskyModel{T}(; Re)
    qr = QRModel{T}(; Re)

    models = [lam, ml, smag, qr]

    for viscosity_model ∈ models
        @testset "$(typeof(viscosity_model))" begin
            setup = Setup{T,2}(;
                viscosity_model,
                grid,
                force,
                pressure_solver,
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

            problem = UnsteadyProblem(setup, V₀, p₀, tlims)
            V, p = solve(problem, RK44(); Δt = 0.01, processors)

            # Check that the average velocity is smaller than the lid velocity
            broken = viscosity_model isa Union{QRModel,MixingLengthModel}
            @test sum(abs, V) / length(V) < lid_vel broken = broken

            # Check for steady state convergence
            broken = viscosity_model isa Union{QRModel,MixingLengthModel}
            @test tracer.umom[end] < 1e-10 broken = broken
            @test tracer.vmom[end] < 1e-10 broken = broken
        end
    end
end
