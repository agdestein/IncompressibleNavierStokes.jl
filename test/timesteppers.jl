@testitem "Time steppers" begin
    using Random
    ax = range(0, 1, 17)
    bc = PeriodicBC(), PeriodicBC()
    setup = Setup(; x = (ax, ax), boundary_conditions = (; u = (bc, bc), temp = (bc, bc)))
    psolver = default_psolver(setup)
    u = random_field(setup)
    temp = randn!(scalarfield(setup))
    temp = apply_bc_temp(temp, 0.0, setup)
    Δt = 0.1
    params = (;
        viscosity = 1e-3,
        conductivity = 1e-3,
        gdir = 2,
        gravity = 1.0,
        dodissipation = true,
    )
    for method in [LMWray3(), RKMethods.RK44()]
        stepper_outplace = let
            state = (; u = copy(u), temp = copy(temp))
            stepper = create_stepper(method; setup, psolver, state, t = 0.0)
            timestep(method, boussinesq, stepper, Δt; params)
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
                params,
            )
        end
        @test stepper_inplace.state.u ≈ stepper_outplace.state.u
        @test stepper_inplace.state.temp ≈ stepper_outplace.state.temp
    end
end

@testitem "Stage times with time-dependent boundary conditions" begin
    # Pulsating plug flow: uniform pulsating inflow u = (uin(t), 0) through a
    # straight channel with symmetric walls. The exact solution is spatially
    # uniform, u = (uin(t), 0) at all times, since the pressure projection
    # forces the uniform interior to match the instantaneous inflow. Any
    # method that applies boundary conditions and projection at the correct
    # stage times reproduces this to machine precision, while incorrect stage
    # times leave an O(Δt) velocity error and an O(Δt) divergence residual in
    # the inflow cells.
    uin(dim, x, y, t) = dim == 1 ? 1 + 0.5 * sinpi(2t) : zero(x)
    boundary_conditions =
        (; u = ((DirichletBC(uin), PressureBC()), (SymmetricBC(), SymmetricBC())))
    setup = Setup(; x = (range(0.0, 4.0, 65), range(0.0, 1.0, 33)), boundary_conditions)
    psolver = default_psolver(setup)
    u0 = vectorfield(setup)
    u0[:, :, 1] .= uin(1, 0.0, 0.0, 0.0)
    u0 = apply_bc_u(u0, 0.0, setup)
    Δt = 0.005
    tend = 0.9
    params = (; viscosity = 0.05)
    uerror(u, t) = maximum(abs, view(u, setup.Iu[1], 1) .- uin(1, 0.0, 0.0, t))
    diverror(u) = maximum(abs, view(divergence(u, setup), setup.Ip))
    for method in [LMWray3(), RKMethods.RK44()]
        state, _ = solve_unsteady(;
            setup,
            psolver,
            tlims = (0.0, tend),
            start = (; u = u0),
            Δt,
            method,
            params,
        )
        @test uerror(state.u, tend) < 1e-10
        @test diverror(state.u) < 1e-10
    end

    # The out-of-place code path of LMWray3 (not exercised by solve_unsteady)
    let method = LMWray3()
        stepper = create_stepper(method; setup, psolver, state = (; u = copy(u0)), t = 0.0)
        nstep = round(Int, tend / Δt)
        for _ = 1:nstep
            stepper = timestep(method, navierstokes, stepper, Δt; params)
        end
        @test stepper.t ≈ tend
        @test uerror(stepper.state.u, tend) < 1e-10
        @test diverror(stepper.state.u) < 1e-10
    end
end
