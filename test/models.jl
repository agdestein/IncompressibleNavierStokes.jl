@testset "Models" begin
    lid_vel = 1.0 # Lid velocity
    u_bc(x, y, t) = y ≈ 1.0 ? lid_vel : 0.0
    v_bc(x, y, t) = 0.0
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    )

    x = cosine_grid(0.0, 1.0, 25)
    y = cosine_grid(0.0, 1.0, 25)

    initial_velocity_u(x, y) = 0.0
    initial_velocity_v(x, y) = 0.0
    initial_pressure(x, y) = 0.0

    # Time interval
    t_start, t_end = tlims = (0.0, 0.5)

    # Iteration processors
    tracer = QuantityTracer()
    processors = [tracer]

    # Viscosity models
    Re = 1000.0
    lam = LaminarModel(; Re)
    kϵ = KEpsilonModel(; Re)
    ml = MixingLengthModel(; Re)
    smag = SmagorinskyModel(; Re)
    qr = QRModel(; Re)

    # Convection models
    noreg = NoRegConvectionModel()
    c2 = C2ConvectionModel()
    c4 = C4ConvectionModel()
    leray = LerayConvectionModel()

    models = [
        (lam, noreg)
        (ml, noreg)
        (smag, noreg)
        (qr, noreg)
        (lam, c2)
        (lam, c4)
    ]

    for (viscosity_model, convection_model) in models
        @testset "$(typeof(viscosity_model)) $(typeof(convection_model))" begin
            setup = Setup(a, b; viscosity_model, convection_model, u_bc, v_bc, bc_type)

            V₀, p₀ = create_initial_conditions(
                setup,
                t_start;
                initial_velocity_u,
                initial_velocity_v,
                initial_pressure,
            )

            problem = SteadyStateProblem(setup, V₀, p₀)
            V, p = solve(problem)

            # Check that the average velocity is smaller than the lid velocity
            broken = convection_model isa Union{C2ConvectionModel,C4ConvectionModel}
            @test sum(abs, V) / length(V) < lid_vel broken = broken

            problem = UnsteadyProblem(setup, V₀, p₀, tlims)
            V, p = solve(problem, RK44(); Δt = 0.01, processors)

            # Check that the average velocity is smaller than the lid velocity
            broken =
                viscosity_model isa Union{QRModel,MixingLengthModel} ||
                convection_model isa Union{C2ConvectionModel,C4ConvectionModel}
            @test sum(abs, V) / length(V) < lid_vel broken = broken

            # Check for steady state convergence
            broken = viscosity_model isa Union{QRModel,MixingLengthModel}
            @test tracer.umom[end] < 1e-10 broken = broken
            @test tracer.vmom[end] < 1e-10 broken = broken
        end
    end

    unfinished_models = [(kϵ, noreg), (lam, leray)]

    for (viscosity_model, convection_model) in models
        @testset "$(typeof(viscosity_model)) $(typeof(convection_model))" begin
            setup = Setup(a, b; viscosity_model, convection_model, u_bc, v_bc, bc_type)
        end
    end
end
