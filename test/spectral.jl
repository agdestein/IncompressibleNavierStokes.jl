@testitem "Energy spectrum" begin
    # Taylor-Green vortices have all their energy in the shell κ = 1
    for D in (2, 3)
        ax = range(0.0, 2π, 33)
        setup = Setup(;
            x = ntuple(Returns(ax), D),
            boundary_conditions = (; u = ntuple(Returns((PeriodicBC(), PeriodicBC())), D)),
        )
        if D == 2
            ufunc = (α, x, y) -> α == 1 ? -sin(x) * cos(y) : cos(x) * sin(y)
            etotal = 1 / 4
        else
            ufunc =
                (α, x, y, z) ->
                    α == 1 ? sin(x) * cos(y) * cos(z) :
                    α == 2 ? -cos(x) * sin(y) * cos(z) : 0.0
            etotal = 1 / 8
        end
        u = velocityfield(setup, ufunc; doproject = false)
        (; κ, ehat) = energyspectrum(u, setup)
        @test κ == 0:16
        @test ehat[2] ≈ etotal # Shell κ = 1
        @test sum(ehat) ≈ etotal
        @test abs(ehat[1]) < 1e-12
        @test all(e -> abs(e) < 1e-12, ehat[3:end])
    end
end

@testitem "Random field" begin
    using Random
    rng = Xoshiro(123)
    for D in (2, 3)
        T = Float64
        n = D == 2 ? 64 : 32
        ax = range(T(0), T(1), n + 1)
        setup = Setup(;
            x = ntuple(Returns(ax), D),
            boundary_conditions = (; u = ntuple(Returns((PeriodicBC(), PeriodicBC())), D)),
        )
        (; Ip, Np) = setup
        totalenergy = T(0.3)
        kpeak = 5
        u = random_field(setup, T(0); totalenergy, kpeak, rng)
        @test eltype(u) == T

        # Exactly divergence free on the staggered grid
        div = divergence(u, setup)
        @test maximum(abs, view(div, Ip)) < 1e-8

        # Spectrum matches the target profile in every shell
        kdiag = isqrt(sum(α -> (Np[α] ÷ 2)^2, 1:D))
        stuff = spectral_stuff(setup; kmax = kdiag)
        (; κ, ehat) = energyspectrum(u, setup; stuff)
        target = map(κ -> orlandi_profile(T(κ); kpeak), κ)
        target = totalenergy .* target ./ sum(target)
        @test sum(ehat) ≈ totalenergy
        @test ehat ≈ target
    end

    # Requires uniform periodic grid
    ax = range(0.0, 1.0, 17)
    setup = Setup(;
        x = (ax, ax),
        boundary_conditions = (;
            u = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC()))
        ),
    )
    @test_throws AssertionError random_field(setup)
end

@testitem "Turbulence statistics" begin
    using Random
    rng = Xoshiro(42)
    viscosity = 1e-3
    ax = range(0.0, 1.0, 33)
    for D in (2, 3)
        setup = Setup(;
            x = ntuple(Returns(ax), D),
            boundary_conditions = (; u = ntuple(Returns((PeriodicBC(), PeriodicBC())), D)),
        )
        totalenergy = 0.1
        u = random_field(setup; totalenergy, rng)
        stats = turbulence_statistics(u, setup, viscosity)
        @test stats isa NamedTuple
        @test all(>(0), values(stats))
        @test stats.e ≈ totalenergy
        if D == 2
            @test haskey(stats, :enstrophy)
            @test haskey(stats, :l_kra)
            @test !haskey(stats, :l_kol)
            @test stats.uavg ≈ sqrt(stats.e)
            @test stats.diss ≈ 2 * viscosity * stats.enstrophy
            @test stats.enstrophy_diss ≈ 2 * viscosity * stats.palinstrophy
        else
            @test haskey(stats, :l_kol)
            @test !haskey(stats, :enstrophy)
            @test stats.uavg ≈ sqrt(2 * stats.e / 3)
            @test stats.Re_tay ≈ stats.l_tay * stats.uavg / viscosity
        end
    end

    # Requires uniform periodic grid
    setup = Setup(;
        x = (ax, ax),
        boundary_conditions = (;
            u = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC()))
        ),
    )
    u = randn!(vectorfield(setup))
    @test_throws AssertionError turbulence_statistics(u, setup, viscosity)
end
