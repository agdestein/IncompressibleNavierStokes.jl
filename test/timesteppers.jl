@testitem "LMWray3" begin
    using Random
    ax = range(0, 1, 17)
    temperature = temperature_equation(;
        Pr = 0.71,
        Ra = 1e7,
        Ge = 1.0,
        boundary_conditions = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
    )
    setup = Setup(; x = (ax, ax), Re = 1e3, temperature)
    psolver = default_psolver(setup)
    method = LMWray3()
    cache = IncompressibleNavierStokes.ode_method_cache(method, setup)
    u = random_field(setup, psolver)
    temp = randn!(scalarfield(setup))
    temp = apply_bc_temp(temp, 0.0, setup)
    Δt = 0.1
    stepper_outplace = let
        stepper = create_stepper(method; setup, psolver, u, temp, t = 0.0)
        timestep(method, stepper, Δt)
    end
    stepper_inplace = let
        stepper = create_stepper(method; setup, psolver, u, temp, t = 0.0)
        IncompressibleNavierStokes.timestep!(method, stepper, Δt; cache)
    end
    @test stepper_inplace.u ≈ stepper_outplace.u
    @test stepper_inplace.temp ≈ stepper_outplace.temp
end
