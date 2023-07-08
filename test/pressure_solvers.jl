@testset "Pressure solvers" begin
    n = 20
    x = LinRange(0, 2π, n)
    y = LinRange(0, 2π, n)
    setup = Setup(x, y)
    (; A) = setup.operators

    # Pressure solvers
    direct = DirectPressureSolver(setup)
    cg = CGPressureSolver(setup)
    spectral = SpectralPressureSolver(setup)

    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    p_exact = reshape(initial_pressure.(setup.grid.xpp, setup.grid.ypp), :)
    f = A * p_exact

    p_direct = pressure_poisson(direct, f)
    p_cg = pressure_poisson(cg, f)
    p_spectral = pressure_poisson(spectral, f)

    # Test that in-place and out-of-place versions give same result
    @test p_direct ≈ pressure_poisson!(direct, zero(p_exact), f)
    @test p_cg ≈ pressure_poisson!(cg, zero(p_exact), f)
    @test p_spectral ≈ pressure_poisson!(spectral, zero(p_exact), f)

    # Test that solvers compute the exact pressure
    @test_broken p_direct ≈ p_exact # `A` is really badly conditioned
    @test p_cg ≈ p_exact
    @test p_spectral ≈ p_exact
end
