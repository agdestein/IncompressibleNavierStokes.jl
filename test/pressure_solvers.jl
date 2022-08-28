@testset "Pressure solvers" begin
    T = Float64

    # Viscosity model
    viscosity_model = LaminarModel{T}(; Re = 1000)

    # Convection model
    convection_model = NoRegConvectionModel()

    # Boundary conditions
    u_bc(x, y, t) = 0.0
    v_bc(x, y, t) = 0.0
    boundary_conditions = BoundaryConditions(
        u_bc,
        v_bc;
        bc_unsteady = false,
        bc_type = (;
            u = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
            v = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        ),
        T,
    )

    # Grid
    x = stretched_grid(0, 2π, 20)
    y = stretched_grid(0, 2π, 20)
    grid = Grid(x, y; boundary_conditions, T)

    # Forcing parameters
    bodyforce_u(x, y) = 0.0
    bodyforce_v(x, y) = 0.0
    force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

    # Build setup and assemble operators
    setup = Setup(; viscosity_model, convection_model, grid, force, boundary_conditions)
    (; A) = setup.operators

    # Pressure solvers
    direct = DirectPressureSolver(setup)
    cg = CGPressureSolver(setup)
    fourier = FourierPressureSolver(setup)

    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    p_exact = reshape(initial_pressure.(grid.xpp, grid.ypp), :)
    f = A * p_exact

    p_direct = IncompressibleNavierStokes.pressure_poisson(direct, f)
    p_cg = IncompressibleNavierStokes.pressure_poisson(cg, f)
    p_fourier = IncompressibleNavierStokes.pressure_poisson(fourier, f)

    # Test that in-place and out-of-place versions give same result
    @test p_direct ≈ IncompressibleNavierStokes.pressure_poisson!(direct, zero(p_exact), f)
    @test p_cg ≈ IncompressibleNavierStokes.pressure_poisson!(cg, zero(p_exact), f)
    @test p_fourier ≈
          IncompressibleNavierStokes.pressure_poisson!(fourier, zero(p_exact), f)

    # Test that solvers compute the exact pressure
    @test_broken p_direct ≈ p_exact # `A` is really badly conditioned
    @test p_cg ≈ p_exact
    @test p_fourier ≈ p_exact
end
