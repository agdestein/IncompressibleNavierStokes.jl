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
#     psolver = DirectPressureSolver(setup)
#     u = random_field(setup, T(0); psolver)
#     randn!.(u)
#     p = randn!(similar(u[1]))
#     test_rrule(apply_bc_u, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
#     test_rrule(apply_bc_p, p, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
# end;

testchainrules(dim) = @testset "Chain rules $(dim())D" begin
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
    setup = Setup(x...; Re)
    psolver = DirectPressureSolver(setup)
    u = random_field(setup, T(0); psolver)
    randn!.(u)
    p = randn!(similar(u[1]))
    @testset "Boundary conditions" begin
        test_rrule(apply_bc_u, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
        test_rrule(apply_bc_p, p, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
    end
    @testset "Divergence" begin
        test_rrule(divergence, u, setup ⊢ NoTangent())
    end
    @testset "Pressure gradient" begin
        test_rrule(pressuregradient, p, setup ⊢ NoTangent())
    end
    @testset "Poisson" begin
        test_rrule(poisson, psolver ⊢ NoTangent(), p)
    end
    @testset "Convection" begin
        test_rrule(convection, u, setup ⊢ NoTangent())
    end
    @testset "Diffusion" begin
        test_rrule(diffusion, u, setup ⊢ NoTangent())
    end
    @testset "Bodyforce" begin
        @test_broken 1 == 2 # Just to identify location for broken rrule test
        # test_rrule(bodyforce, u, T(0) ⊢ NoTangent(), setup ⊢ NoTangent())
    end
end

testchainrules(IncompressibleNavierStokes.Dimension(2))
testchainrules(IncompressibleNavierStokes.Dimension(3));
