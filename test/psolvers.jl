@testitem "Pressure solvers" begin
    n = 32
    ax = LinRange(0, 2π, n + 1)
    setup = Setup(;
        x = (ax, ax),
        boundary_conditions = (;
            u = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
        ),
    )
    (; Ip, xp) = setup
    D = 2

    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    p_exact = initial_pressure.(
        ntuple(α -> reshape(xp[α], ntuple(Returns(1), α - 1)..., :), D)...,
    )
    IncompressibleNavierStokes.apply_bc_p!(p_exact, 0.0, setup)
    lap = IncompressibleNavierStokes.laplacian(p_exact, setup)

    # Pressure solvers
    direct = psolver_direct(setup)
    cg = psolver_cg(setup)
    spectral = psolver_spectral(setup)
    transform = psolver_transform(setup)

    get_p(psolver) = IncompressibleNavierStokes.apply_bc_p(
        IncompressibleNavierStokes.poisson(psolver, lap),
        0.0,
        setup,
    )

    # Test that solvers compute the exact pressure
    @test get_p(direct)[Ip] ≈ p_exact[Ip]
    @test get_p(cg)[Ip] ≈ p_exact[Ip]
    @test get_p(spectral)[Ip] ≈ p_exact[Ip]
    @test get_p(transform)[Ip] ≈ p_exact[Ip]
end

@testitem "Tridiagonal pressure solver" begin
    using Random
    using Statistics

    # Channel-like setups: walls in direction `dir`, periodic elsewhere.
    # The wall-normal grid is stretched, the periodic directions are uniform.
    for D in (2, 3), dir = 1:D
        n = 16
        x = ntuple(D) do β
            β == dir ? cosine_grid(0.0, 1.0, n) : LinRange(0.0, 2π, 2n + 1)
        end
        boundary_conditions = (;
            u = ntuple(D) do β
                β == dir ? (DirichletBC(), DirichletBC()) : (PeriodicBC(), PeriodicBC())
            end,
        )
        setup = Setup(; x, boundary_conditions)
        (; Ip) = setup

        # Manufactured pressure with the right hand side from its Laplacian
        rng = Xoshiro(129 + D + dir)
        p_exact = IncompressibleNavierStokes.scalarfield(setup)
        view(p_exact, Ip) .= randn(rng, size(Ip))
        IncompressibleNavierStokes.apply_bc_p!(p_exact, 0.0, setup)
        lap = IncompressibleNavierStokes.laplacian(p_exact, setup)

        tridiagonal = psolver_tridiagonal(setup)
        p = IncompressibleNavierStokes.poisson(tridiagonal, lap)

        # Pressure is only determined up to a constant
        @test p[Ip] .- mean(p[Ip]) ≈ p_exact[Ip] .- mean(p_exact[Ip])

        # The wall-normal direction is inferred, but can also be given
        explicit = psolver_tridiagonal(setup; dir)
        @test explicit(copy(lap))[Ip] ≈ p[Ip]

        # The chosen direction must be wall-bounded
        wrongdir = mod1(dir + 1, D)
        @test_throws ErrorException psolver_tridiagonal(setup; dir = wrongdir)

        # This setup is also chosen by default
        # (the closure name is not portable across Julia versions,
        # so check for a captured variable instead)
        @test :fftscale in propertynames(default_psolver(setup))
    end

    # Fully periodic setups have no wall-normal direction
    ax = LinRange(0.0, 1.0, 17)
    setup = Setup(;
        x = (ax, ax),
        boundary_conditions = (;
            u = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
        ),
    )
    @test_throws ErrorException psolver_tridiagonal(setup)
end
