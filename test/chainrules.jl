"Use function name only as test set name"
test_rrule_named(f, args...; kwargs...) =
    test_rrule(f, args...; testset_name = string(f), kwargs...)

# @testset "Chain rules boundary conditions" begin
#     T = Float64
#     Re = T(1_000)
#     n = 8
#     boundary_conditions = ((DirichletBC(), PressureBC()), (PeriodicBC(), PeriodicBC()))
#     lims = T(0), T(1)
#     x = stretched_grid(lims..., n, 1.2), cosine_grid(lims..., n)
#     setup = Setup(x...; Re,
#         boundary_conditions,
#     )
#     u = random_field(setup, T(0))
#     randn!.(u)
#     p = randn!(similar(u[1]))
#     test_rrule_named(apply_bc_u, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
#     test_rrule_named(apply_bc_p, p, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
# end;

# Test chain rule correctness by comparing with finite differences
testchainrules(dim) = @testset "Chain rules $(dim())D" begin
    @info "Testing chain rules in $(dim())D"

    # Setup
    D = dim()
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
        stretched_grid(lims..., n, 1.2), cosine_grid(lims..., n)
    elseif D == 3
        stretched_grid(lims..., n, 1.2), cosine_grid(lims..., n), cosine_grid(lims..., n)
    end
    temperature = temperature_equation(;
        Pr = T(0.71),
        Ra = T(1e6),
        Ge = T(1.0),
        dodissipation = true,
        boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
        gdir = 2,
        nondim_type = 1,
    )
    setup = Setup(x...; Re, temperature)
    psolver = default_psolver(setup)
    u = random_field(setup, T(0); psolver)
    randn!.(u)
    p = randn!(similar(u[1]))
    temp = randn!(similar(u[1]))

    # Tests
    test_rrule_named(apply_bc_u, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
    test_rrule_named(apply_bc_p, p, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
    test_rrule_named(apply_bc_temp, temp, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
    test_rrule_named(divergence, u, setup ⊢ NoTangent())
    test_rrule_named(pressuregradient, p, setup ⊢ NoTangent())
    test_rrule_named(poisson, psolver ⊢ NoTangent(), p)
    test_rrule_named(convection, u, setup ⊢ NoTangent())
    test_rrule_named(diffusion, u, setup ⊢ NoTangent())

    @test_broken 1 == 2 # Just to identify location for broken rrule test
    # test_rrule_named(bodyforce, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())

    test_rrule_named(gravity, temp, setup ⊢ NoTangent())

    @test_broken 1 == 2 # Just to identify location for broken rrule test
    # test_rrule_named(dissipation, u, setup ⊢ NoTangent())
    # ChainRulesCore.rrule(dissipation, u, setup)[2](temp)[2][2]

    @test_broken 1 == 2 # Just to identify location for broken rrule test
    # test_rrule_named(convection_diffusion_temp, u, temp, setup ⊢ NoTangent())
end

testchainrules(IncompressibleNavierStokes.Dimension(2));
testchainrules(IncompressibleNavierStokes.Dimension(3));
