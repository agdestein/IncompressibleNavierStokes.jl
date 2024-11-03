@testsnippet ChainRulesStuff begin
    using ChainRulesCore
    using ChainRulesTestUtils

    "Use function name only as test set name"
    test_rrule_named(f, args...; kwargs...) =
        test_rrule(f, args...; testset_name = string(f), kwargs...)
end

@testitem "Chain rules (boundary conditions)" setup = [ChainRulesStuff] begin
    T = Float64
    Re = T(1_000)
    Pr = T(0.71)
    Ra = T(1e6)
    Ge = T(1.0)
    n = 7
    lims = T(0), T(1)
    x = range(lims..., n + 1), range(lims..., n + 1)
    for bc in (PeriodicBC(), DirichletBC(), SymmetricBC(), PressureBC())
        boundary_conditions = ((bc, bc), (bc, bc))
        setup = Setup(;
            x,
            Re,
            boundary_conditions,
            temperature = temperature_equation(; Pr, Ra, Ge, boundary_conditions),
        )
        u = randn(T, setup.grid.N), randn(T, setup.grid.N)
        p = randn(T, setup.grid.N)
        temp = randn(T, setup.grid.N)
        test_rrule_named(apply_bc_u, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
        test_rrule_named(apply_bc_p, p, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
        test_rrule_named(apply_bc_temp, temp, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
    end
end

@testmodule Case begin
    using IncompressibleNavierStokes

    D2, D3 = map((2, 3)) do D
        T = Float64
        Re = T(1_000)
        n = if D == 2
            8
        elseif D == 3
            # 4^3 = 64 grid points
            # 3*64 = 192 velocity components
            # 192^2 = 36864 fininite difference pairs in convection/diffusion
            # TODO: Check if `test_rrule` computes all combinations or only a subset
            4
        end
        lims = T(0), T(1)
        x = if D == 2
            tanh_grid(lims..., n), tanh_grid(lims..., n, 1.3)
        elseif D == 3
            tanh_grid(lims..., n, 1.2), tanh_grid(lims..., n, 1.1), cosine_grid(lims..., n)
        end
        boundary_conditions = ntuple(d -> (DirichletBC(), DirichletBC()), D)
        temperature = temperature_equation(;
            Pr = T(0.71),
            Ra = T(1e6),
            Ge = T(1.0),
            boundary_conditions,
        )
        setup = Setup(; x, boundary_conditions, Re, temperature)
        psolver = default_psolver(setup)
        u = ntuple(α -> randn(T, setup.grid.N), D)
        p = randn(T, setup.grid.N)
        temp = randn(T, setup.grid.N)
        (; setup, psolver, u, p, temp)
    end
end

@testitem "divergence" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(divergence, Case.D2.u, Case.D2.setup ⊢ NoTangent())
    test_rrule_named(divergence, Case.D3.u, Case.D3.setup ⊢ NoTangent())
end

@testitem "pressuregradient" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(pressuregradient, Case.D2.p, Case.D2.setup ⊢ NoTangent())
    test_rrule_named(pressuregradient, Case.D3.p, Case.D3.setup ⊢ NoTangent())
end

@testitem "poisson" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(poisson, Case.D2.psolver ⊢ NoTangent(), Case.D2.p)
    test_rrule_named(poisson, Case.D3.psolver ⊢ NoTangent(), Case.D3.p)
end

@testitem "convection" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(convection, Case.D2.u, Case.D2.setup ⊢ NoTangent())
    test_rrule_named(convection, Case.D3.u, Case.D3.setup ⊢ NoTangent())
end

@testitem "diffusion" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(diffusion, Case.D2.u, Case.D2.setup ⊢ NoTangent())
    test_rrule_named(diffusion, Case.D3.u, Case.D3.setup ⊢ NoTangent())
end

@testitem "Chain rules (operators)" setup = [Case, ChainRulesStuff] begin
    # Test chain rule correctness by comparing with finite differences

    @test_broken 1 == 2 # Just to identify location for broken rrule test
    # test_rrule_named(bodyforce, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())

    test_rrule_named(gravity, temp, setup ⊢ NoTangent())

    @test_broken 1 == 2 # Just to identify location for broken rrule test
    # test_rrule_named(dissipation, u, setup ⊢ NoTangent())
    # ChainRulesCore.rrule(dissipation, u, setup)[2](temp)[2][2]

    @test_broken 1 == 2 # Just to identify location for broken rrule test
    # test_rrule_named(convection_diffusion_temp, u, temp, setup ⊢ NoTangent())
end
