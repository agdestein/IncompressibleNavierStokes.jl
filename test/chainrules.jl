@testsnippet ChainRulesStuff begin
    using ChainRulesCore
    using ChainRulesTestUtils

    # Test chain rule correctness by comparing with finite differences
    "Use function name only as test set name"
    function test_rrule_named(f, args...; kwargs...)
        test_rrule(f, args...; testset_name = string(f), kwargs...)
    end
end

@testitem "Chain rules (boundary conditions)" setup = [ChainRulesStuff] begin
    T = Float64
    params = (
        viscosity = T(1e-3),
        conductivity = T(1e-3),
        gdir = 2,
        gravity = T(1.0),
        dodissipation = true,
    )
    n = 7
    lims = T(0), T(1)
    ax = range(lims..., n + 1)
    x = ax, ax
    for bc in (PeriodicBC(), DirichletBC(), SymmetricBC(), PressureBC())
        BC = (bc, bc), (bc, bc)
        setup = Setup(; x, boundary_conditions = (; u = BC, temp = BC))
        u = randn(T, setup.N..., 2)
        p = randn(T, setup.N)
        temp = randn(T, setup.N)
        test_rrule_named(apply_bc_u, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
        test_rrule_named(apply_bc_p, p, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
        test_rrule_named(apply_bc_temp, temp, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
    end
end

@testmodule Case begin
    # module Case
    using IncompressibleNavierStokes

    D2, D3 = map((2, 3)) do D
        T = Float64
        params = (
            viscosity = T(1e-3),
            conductivity = T(1e-3),
            gdir = 2,
            gravity = T(1.0),
            dodissipation = true,
        )
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
        bc = ntuple(d -> (DirichletBC(), DirichletBC()), D)
        setup = Setup(; x, boundary_conditions = (; u = bc, temp = bc))
        psolver = default_psolver(setup)
        u = randn(T, setup.N..., D)
        p = randn(T, setup.N)
        temp = randn(T, setup.N)
        (; setup, psolver, u, p, temp, params)
    end
end

@testitem "Divergence" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(divergence, Case.D2.u, Case.D2.setup ⊢ NoTangent())
    test_rrule_named(divergence, Case.D3.u, Case.D3.setup ⊢ NoTangent())
end

@testitem "Pressuregradient" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(pressuregradient, Case.D2.p, Case.D2.setup ⊢ NoTangent())
    test_rrule_named(pressuregradient, Case.D3.p, Case.D3.setup ⊢ NoTangent())
end

@testitem "Poisson" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(poisson, Case.D2.psolver ⊢ NoTangent(), Case.D2.p)
    test_rrule_named(poisson, Case.D3.psolver ⊢ NoTangent(), Case.D3.p)
end

@testitem "Convection" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(convection, Case.D2.u, Case.D2.setup ⊢ NoTangent())
    test_rrule_named(convection, Case.D3.u, Case.D3.setup ⊢ NoTangent())
end

# # Convection
# let
#     using Random
#     using LinearAlgebra
#     # w = rand!(similar(Case.D2.u))
#     w = zero(Case.D2.u)
#     w[end-1, 4, 1] = 1
#     u = copy(Case.D2.u)
#     g_fd = map(u |> axes |> CartesianIndices) do I
#         h = u |> eltype |> eps |> cbrt
#         a = u |> copy
#         b = u |> copy
#         a[I] -= h / 2
#         b[I] += h / 2
#         fa = convection(a, Case.D2.setup)
#         fb = convection(b, Case.D2.setup)
#         (dot(w .* fb, fb) - dot(w .* fa, fa)) / 2 / h
#         # (dot(fb, fb) - dot(fa, fa)) / 2 / h
#     end;
#     c, c_pb = ChainRulesCore.rrule(convection, u, Case.D2.setup);
#     c_pb(w .* c)[2][:, :, 1]
#     g_fd[:, :, 1]
#     c_pb(w .* c)[2] - g_fd
# end

# # Diffusion
# let
#     viscosity = Case.D2.params.viscosity
#     g_fd = map(Case.D2.u |> axes |> CartesianIndices) do I
#         h = Case.D2.u |> eltype |> eps |> cbrt
#         a = Case.D2.u |> copy
#         b = Case.D2.u |> copy
#         a[I] -= h / 2
#         b[I] += h / 2
#         fa = diffusion(a, Case.D2.setup, viscosity)
#         fb = diffusion(b, Case.D2.setup, viscosity)
#         (sum(abs2, fb) - sum(abs2, fa)) / 2 / h
#     end
#     c, c_pb = ChainRulesCore.rrule(diffusion, Case.D2.u, Case.D2.setup, viscosity)
#     c_pb(c)[2]
#     g_fd
#     c
#     c_pb(c)[2] - g_fd
# end

@testitem "Diffusion" setup = [Case, ChainRulesStuff] begin
    test_rrule_named(
        diffusion,
        Case.D2.u,
        Case.D2.setup ⊢ NoTangent(),
        Case.D2.params.viscosity ⊢ NoTangent(),
    )
    test_rrule_named(
        diffusion,
        Case.D3.u,
        Case.D3.setup ⊢ NoTangent(),
        Case.D2.params.viscosity ⊢ NoTangent(),
    )
end

@testitem "Temperature" setup = [Case, ChainRulesStuff] begin
    case = Case.D2
    for case in (Case.D2, Case.D3)
        (; u, temp, setup) = case

        test_rrule_named(
            applygravity,
            temp,
            setup ⊢ NoTangent(),
            Case.D2.params.gdir ⊢ NoTangent(),
            Case.D2.params.gravity ⊢ NoTangent(),
        )

        @test_broken 1 == 2 # Just to identify location for broken rrule test
        # test_rrule_named(dissipation, u, setup ⊢ NoTangent())
        # ChainRulesCore.rrule(dissipation, u, setup)[2](temp)[2][2]

        @test_broken 1 == 2 # Just to identify location for broken rrule test
        # test_rrule_named(convection_diffusion_temp, u, temp, setup ⊢ NoTangent())
    end
end
