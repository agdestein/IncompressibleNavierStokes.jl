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
    lmspectral = LowMemorySpectralPressureSolver(setup)

    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    p_exact =
        initial_pressure.(
            ntuple(α -> reshape(xp[α], ntuple(Returns(1), α - 1)..., :), D)...,
        )
    IncompressibleNavierStokes.apply_bc_p!(p_exact, 0.0, setup)
    lap = IncompressibleNavierStokes.laplacian(p_exact, setup)

    get_p(psolver) = IncompressibleNavierStokes.apply_bc_p(
        IncompressibleNavierStokes.poisson(psolver, lap),
        0.0,
        setup,
    )
    p_direct = get_p(direct)
    p_cg = get_p(cg)
    p_spectral = get_p(spectral)
    p_lmspectral = get_p(lmspectral)

    # Test that in-place and out-of-place versions give same result
    get_p_inplace(psolver) = IncompressibleNavierStokes.apply_bc_p!(
        IncompressibleNavierStokes.poisson!(psolver, zero(p_exact), lap),
        0.0,
        setup,
    )
    @test p_direct ≈ get_p_inplace(direct)
    @test p_cg ≈ get_p_inplace(cg)
    @test p_spectral ≈ get_p_inplace(spectral)
    @test p_lmspectral ≈ get_p_inplace(lmspectral)

    # Test that solvers compute the exact pressure
    @test p_direct ≈ p_exact
    @test p_cg ≈ p_exact
    @test p_spectral ≈ p_exact
    @test p_lmspectral ≈ p_exact
end
