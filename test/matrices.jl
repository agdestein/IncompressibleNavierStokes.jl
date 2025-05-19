@testsnippet Matrices begin
    using Random
    x = tanh_grid(0.0, 5.0, 11), cosine_grid(0.0, 1.0, 7), tanh_grid(0.0, 0.8, 5)
    setup = Setup(;
        x,
        visc = 1e-3,
        boundary_conditions = (
            (PeriodicBC(), PeriodicBC()),
            (DirichletBC(), PressureBC()),
            (SymmetricBC(), SymmetricBC()),
        ),
    )
    u = randn!(vectorfield(setup))
    p = randn!(scalarfield(setup))
end

@testitem "BC matrix" setup = [Matrices] begin
    # Test that the two BC variants give the same results
    Bu = IncompressibleNavierStokes.bc_u_mat(setup)
    Bp = IncompressibleNavierStokes.bc_p_mat(setup)
    u1 = reshape(Bu * u[:], size(u))
    p1 = reshape(Bp * p[:], size(p))
    u2 = apply_bc_u(u, 0, setup)
    p2 = apply_bc_p(p, 0, setup)
    @test u1 ≈ u2
    @test p1 ≈ p2
end

@testitem "Divergence matrix" setup = [Matrices] begin
    B = IncompressibleNavierStokes.bc_u_mat(setup)
    M = IncompressibleNavierStokes.divergence_mat(setup)
    div1 = reshape(M * B * u[:], size(p))
    div2 = divergence(apply_bc_u(u, 0, setup), setup)
    @test div1 ≈ div2
end

@testitem "Pressure gradient matrix" setup = [Matrices] begin
    B = IncompressibleNavierStokes.bc_p_mat(setup)
    G = IncompressibleNavierStokes.pressuregradient_mat(setup)
    g1 = reshape(G * B * p[:], size(u))
    g2 = pressuregradient(apply_bc_p(p, 0, setup), setup)
    @test g1 ≈ g2
end

@testitem "Diffusion matrix" setup = [Matrices] begin
    B = IncompressibleNavierStokes.bc_u_mat(setup)
    D = IncompressibleNavierStokes.diffusion_mat(setup)
    d1 = reshape(D * B * u[:], size(u))
    d2 = diffusion(apply_bc_u(u, 0, setup), setup; use_viscosity = false)
    @test d1 ≈ d2
end
