@testset "Pressure solvers" begin
    @info "Testing pressure solvers"
    n = 32
    x = LinRange(0, 2π, n + 1)
    y = LinRange(0, 2π, n + 1)
    Re = 1e3
    setup = Setup(x, y; Re)
    (; xp) = setup.grid
    D = 2

    # Pressure solvers
    direct = DirectPressureSolver(setup)
    cg = CGPressureSolver(setup)
    spectral = SpectralPressureSolver(setup)

    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    p_exact =
        initial_pressure.(
            ntuple(α -> reshape(xp[α], ntuple(Returns(1), α - 1)..., :), D)...,
        )
    IncompressibleNavierStokes.apply_bc_p!(p_exact, 0.0, setup)
    lap = IncompressibleNavierStokes.laplacian(p_exact, setup)

    p_direct = IncompressibleNavierStokes.apply_bc_p!(
        IncompressibleNavierStokes.poisson(direct, lap),
        0.0,
        setup,
    )
    p_cg = IncompressibleNavierStokes.apply_bc_p!(
        IncompressibleNavierStokes.poisson(cg, lap),
        0.0,
        setup,
    )
    p_spectral = IncompressibleNavierStokes.apply_bc_p!(
        IncompressibleNavierStokes.poisson(spectral, lap),
        0.0,
        setup,
    )

    # Test that in-place and out-of-place versions give same result
    @test p_direct ≈ IncompressibleNavierStokes.apply_bc_p!(
        IncompressibleNavierStokes.poisson!(direct, zero(p_exact), lap),
        0.0,
        setup,
    )
    @test p_cg ≈ IncompressibleNavierStokes.apply_bc_p!(
        IncompressibleNavierStokes.poisson!(cg, zero(p_exact), lap),
        0.0,
        setup,
    )
    @test p_spectral ≈ IncompressibleNavierStokes.apply_bc_p!(
        IncompressibleNavierStokes.poisson!(spectral, zero(p_exact), lap),
        0.0,
        setup,
    )

    # Test that solvers compute the exact pressure
    @test p_direct ≈ p_exact
    @test p_cg ≈ p_exact
    @test p_spectral ≈ p_exact
end
