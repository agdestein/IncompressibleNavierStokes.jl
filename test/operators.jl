@testset "Operators" begin
    # Setup
    T = Float64
    Re = T(1_000)
    n = 16
    lims = T(0), T(1)
    x = stretched_grid(lims..., n, 1.2), cosine_grid(lims..., n)
    setup = Setup(x...; Re)
    psolver = DirectPressureSolver(setup)
    u = random_field(setup, T(0); psolver)
    (; Iu, Ip) = setup.grid

    @testset "Divergence" begin
        div = divergence(u, setup)
        @test div isa Array{T}
        @test all(!isnan, div)
    end
    
    @testset "Pressure gradient" begin
        v = randn!.(similar.(u))
        p = randn!(similar(u[1]))
        apply_bc_u!(v, T(0), setup)
        apply_bc_p!(p, T(0), setup)
        Dv = divergence(v, setup)
        Gp = pressuregradient(p, setup)
        pDv = if length(v) == 2
            sum((p .* Ω .* Dv)[Ip])
        end
        vGp = if length(v) == 2
            vGpx = v[1] .* setup.grid.Δu[1] .* setup.grid.Δ[2]' .* Gp[1]
            vGpx = v[2] .* setup.grid.Δ[1] .* setup.grid.Δu[2]' .* Gp[2]
            sum(vGpx[Iu[1]]) + sum(vGpx[Iu[2]])
        end
        @test G isa Tuple
        @test G[1] isa Array{T}
        @test pDv ≈ -vGp # Check that D = -G'
    end

    @testset "Convection" begin
        c = convection(u, setup)
        uCu = if length(u) == 2
            uCux = u[1] .* setup.grid.Δu[1] .* setup.grid.Δ[2]' .* c[1]
            uCuy = u[2] .* setup.grid.Δ[1] .* setup.grid.Δu[2]' .* c[2]
            sum(uCux[Iu[1]]) + sum(uCuy[Iu[2]])
        end
        @test c isa Tuple
        @test c[1] isa Array{T}
        @test uCu ≈ 0 atol = 1e-12 # Check skew-symmetry
    end

    @testset "Diffusion" begin
        d = diffusion(u, setup)
        uDu = if length(u) == 2
            uDux = u[1] .* setup.grid.Δu[1] .* setup.grid.Δ[2]' .* d[1]
            uDuy = u[2] .* setup.grid.Δ[1] .* setup.grid.Δu[2]' .* d[2]
            sum(uDux[Iu[1]]) + sum(uDuy[Iu[2]])
        end
        @test d isa Tuple
        @test d[1] isa Array{T}
        @test uDu ≥ 0 # Check positivity
    end
end
