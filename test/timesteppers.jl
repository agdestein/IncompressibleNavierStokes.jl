@testitem "In/out-of place" begin
    using Random
    ax = range(0, 1, 17)
    temperature = temperature_equation(;
        Pr = 0.71,
        Ra = 1e7,
        Ge = 1.0,
        boundary_conditions = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
    )
    setup = Setup(; x = (ax, ax), visc = 1e-3, temperature)
    psolver = default_psolver(setup)
    u = random_field(setup, psolver)
    temp = randn!(scalarfield(setup))
    temp = apply_bc_temp(temp, 0.0, setup)
    Δt = 0.1
    for method in [LMWray3(), RKMethods.RK44()]
        stepper_outplace = let
            state = (; u = copy(u), temp = copy(temp))
            stepper = create_stepper(method; setup, psolver, state, t = 0.0)
            timestep(method, boussinesq, stepper, Δt)
        end
        stepper_inplace = let
            state = (; u = copy(u), temp = copy(temp))
            force_cache = IncompressibleNavierStokes.get_cache(boussinesq!, setup)
            ode_cache = IncompressibleNavierStokes.get_cache(method, state, setup)
            stepper = create_stepper(method; setup, psolver, state, t = 0.0)
            IncompressibleNavierStokes.timestep!(
                method,
                boussinesq!,
                stepper,
                Δt;
                ode_cache,
                force_cache,
            )
        end
        @test stepper_inplace.state.u ≈ stepper_outplace.state.u
        @test stepper_inplace.state.temp ≈ stepper_outplace.state.temp
    end
end
