@testset "Models" begin
    lid_vel = 1.0 # Lid velocity
    u_bc(x, y, t) = y ≈ 1.0 ? lid_vel : 0.0
    v_bc(x, y, t) = 0.0
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ν = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    )

    n = 25
    x = LinRange(0.0, 1.0, n)
    y = LinRange(0.0, 1.0, n)
    Δ = √2 / n

    initial_velocity_u(x, y) = 0.0
    initial_velocity_v(x, y) = 0.0
    initial_pressure(x, y) = 0.0

    # Time interval
    t_start, t_end = tlims = (0.0, 0.5)

    # Viscosity models
    T = Float64
    Re = 1000.0
    lam = LaminarModel(; Re)
    ml = MixingLengthModel(; Re, lm = Δ)
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
            @info "Testing $(typeof(viscosity_model)) and $(typeof(convection_model))"
            setup = Setup(x, y; viscosity_model, convection_model, u_bc, v_bc, bc_type)

            V₀, p₀ = create_initial_conditions(
                setup,
                initial_velocity_u,
                initial_velocity_v,
                t_start;
                initial_pressure,
            )

            V, p = solve_steady_state(setup, V₀, p₀; npicard = 5, maxiter = 15) # Check that the average velocity is smaller than the lid velocity
            broken = convection_model isa Union{C2ConvectionModel,C4ConvectionModel}
            @test sum(abs, V) / length(V) < lid_vel broken = broken

            V, p, outputs = solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.01)

            # Check that the average velocity is smaller than the lid velocity
            broken = convection_model isa Union{C2ConvectionModel,C4ConvectionModel}
            @test sum(abs, V) / length(V) < lid_vel broken = broken
        end
    end

    unfinished_models = [(lam, leray)]

    for (viscosity_model, convection_model) in models
        @testset "$(typeof(viscosity_model)) $(typeof(convection_model))" begin
            @info "Testing $(typeof(viscosity_model)) and $(typeof(convection_model))"
            setup = Setup(x, y; viscosity_model, convection_model, u_bc, v_bc, bc_type)
        end
    end
end
