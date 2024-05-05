@testset "Pressure solvers" begin
    @info "Testing pressure solvers"
    n = 32
    x = LinRange(0, 2π, n + 1)
    y = LinRange(0, 2π, n + 1)
    Re = 1e3
    setup = Setup(x, y; Re)
    (; xp) = setup.grid
    D = 2

    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    p_exact =
        initial_pressure.(
            ntuple(α -> reshape(xp[α], ntuple(Returns(1), α - 1)..., :), D)...,
        )
    IncompressibleNavierStokes.apply_bc_p!(p_exact, 0.0, setup)
    lap = IncompressibleNavierStokes.laplacian(p_exact, setup)

    # Pressure solvers
    direct = psolver_direct(setup)
    cg = psolver_cg(setup)
    spectral = psolver_spectral(setup)
    spectral_lowmemory = solver_spectral_lowmemory(setup)

    get_p(psolver) = IncompressibleNavierStokes.apply_bc_p(
        IncompressibleNavierStokes.poisson(psolver, lap),
        0.0,
        setup,
    )

    # Test that solvers compute the exact pressure
    @test get_p(p_direct) ≈ p_exact
    @test get_p(p_cg) ≈ p_exact
    @test get_p(p_spectral) ≈ p_exact
    @test get_p(p_spectral_lowmemory) ≈ p_exact
end
