@testitem "Operators" begin
    using Random
    using IncompressibleNavierStokes: convectiondiffusion!
    using LinearAlgebra
    using SparseArrays

    testops(dim) = @testset "Operators $(dim())D" begin
        # Setup
        D = dim()
        T = Float64
        Re = T(1_000)
        n = 16
        lims = T(0), T(1)
        x = if D == 2
            tanh_grid(lims..., n), tanh_grid(lims..., n, 1.3)
        elseif D == 3
            tanh_grid(lims..., n, 1.2), tanh_grid(lims..., n, 1.1), cosine_grid(lims..., n)
        end
        setup = Setup(
            x...;
            Re,
            boundary_conditions = ntuple(d -> (DirichletBC(), DirichletBC()), D),
        )
        uref(dim, x, y, args...) =
            -(dim() == 1) * sin(x) * cos(y) + (dim == 2) * cos(x) * sin(y)
        u = velocityfield(setup, uref, T(0))
        (; Iu, Ip, Ω) = setup.grid

        @testset "Divergence" begin
            div = divergence(u, setup)
            @test div isa Array{T}
            @test all(!isnan, div)
        end

        @testset "Pressure gradient" begin
            v = randn!.(vectorfield(setup))
            p = randn!(scalarfield(setup))
            v = apply_bc_u(v, T(0), setup)
            p = apply_bc_p(p, T(0), setup)
            Dv = divergence(v, setup)
            Gp = pressuregradient(p, setup)
            pDv = sum((p.*Ω.*Dv)[Ip])
            vGp = if D == 2
                vGpx = v[1] .* setup.grid.Δu[1] .* setup.grid.Δ[2]' .* Gp[1]
                vGpy = v[2] .* setup.grid.Δ[1] .* setup.grid.Δu[2]' .* Gp[2]
                sum(vGpx[Iu[1]]) + sum(vGpy[Iu[2]])
            elseif D == 3
                vGpx =
                    v[1] .* setup.grid.Δu[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                    reshape(setup.grid.Δ[3], 1, 1, :) .* Gp[1]
                vGpy =
                    v[2] .* setup.grid.Δ[1] .* reshape(setup.grid.Δu[2], 1, :) .*
                    reshape(setup.grid.Δ[3], 1, 1, :) .* Gp[2]
                vGpz =
                    v[3] .* setup.grid.Δ[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                    reshape(setup.grid.Δu[3], 1, 1, :) .* Gp[3]
                sum(vGpx[Iu[1]]) + sum(vGpy[Iu[2]]) + sum(vGpz[Iu[3]])
            end
            @test Gp isa Tuple
            @test Gp[1] isa Array{T}
            @test pDv ≈ -vGp # Check that D = -G'
        end

        @testset "Laplacian" begin
            p = randn!(scalarfield(setup))
            p = apply_bc_p(p, T(0), setup)
            Lp = laplacian(p, setup)
            @test Lp isa Array{T}
            @test sum((p.*Ω.*Lp)[Ip]) ≤ 0 # Check negativity
            L = laplacian_mat(setup)
            @test L isa SparseMatrixCSC
            @test norm((Lp)[Ip][:] - (L * p[Ip][:])) ≈ 0 atol = 1e-12
        end

        @testset "Convection" begin
            c = convection(u, setup)
            uCu = if D == 2
                uCux = u[1] .* setup.grid.Δu[1] .* setup.grid.Δ[2]' .* c[1]
                uCuy = u[2] .* setup.grid.Δ[1] .* setup.grid.Δu[2]' .* c[2]
                sum(uCux[Iu[1]]) + sum(uCuy[Iu[2]])
            elseif D == 3
                uCux =
                    u[1] .* setup.grid.Δu[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                    reshape(setup.grid.Δ[3], 1, 1, :) .* c[1]
                uCuy =
                    u[2] .* setup.grid.Δ[1] .* reshape(setup.grid.Δu[2], 1, :) .*
                    reshape(setup.grid.Δ[3], 1, 1, :) .* c[2]
                uCuz =
                    u[3] .* setup.grid.Δ[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                    reshape(setup.grid.Δu[3], 1, 1, :) .* c[3]
                sum(uCux[Iu[1]]) + sum(uCuy[Iu[2]]) + sum(uCuz[Iu[3]])
            end
            @test c isa Tuple
            @test c[1] isa Array{T}
            @test uCu ≈ 0 atol = 1e-12 # Check skew-symmetry
        end

        @testset "Diffusion" begin
            d = diffusion(u, setup)
            uDu = if D == 2
                uDux = u[1] .* setup.grid.Δu[1] .* setup.grid.Δ[2]' .* d[1]
                uDuy = u[2] .* setup.grid.Δ[1] .* setup.grid.Δu[2]' .* d[2]
                sum(uDux[Iu[1]]) + sum(uDuy[Iu[2]])
            elseif D == 3
                uDux =
                    u[1] .* setup.grid.Δu[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                    reshape(setup.grid.Δ[3], 1, 1, :) .* d[1]
                uDuy =
                    u[2] .* setup.grid.Δ[1] .* reshape(setup.grid.Δu[2], 1, :) .*
                    reshape(setup.grid.Δ[3], 1, 1, :) .* d[2]
                uDuz =
                    u[3] .* setup.grid.Δ[1] .* reshape(setup.grid.Δ[2], 1, :) .*
                    reshape(setup.grid.Δu[3], 1, 1, :) .* d[3]
                sum(uDux[Iu[1]]) + sum(uDuy[Iu[2]]) + sum(uDuz[Iu[3]])
            end
            @test d isa Tuple
            @test d[1] isa Array{T}
            @test uDu ≤ 0 # Check negativity (dissipation)
        end

        @testset "Convection-Diffusion" begin
            cd = convectiondiffusion!(zero.(u), u, setup)
            c = convection(u, setup)
            d = diffusion(u, setup)
            @test all(cd .≈ c .+ d)
        end

        @testset "Momentum" begin
            m = momentum(u, nothing, T(1), setup)
            @test m isa Tuple
            @test m[1] isa Array{T}
            @test all(all.(!isnan, m))
        end

        @testset "Other fields" begin
            p = randn!(scalarfield(setup))
            ω = vorticity(u, setup)
            D == 2 && @test ω isa Array{T}
            D == 3 && @test ω isa Tuple
            @test smagorinsky_closure(setup)(u, 0.1) isa Tuple
            @test tensorbasis(u, setup) isa Tuple
            @test interpolate_u_p(u, setup) isa Tuple
            D == 2 && @test interpolate_ω_p(ω, setup) isa Array{T}
            D == 3 && @test interpolate_ω_p(ω, setup) isa Tuple
            @test Dfield(p, setup) isa Array{T}
            @test Qfield(u, setup) isa Array{T}
            D == 2 && @test_throws AssertionError eig2field(u, setup)
            D == 3 && @test eig2field(u, setup) isa Array{T} broken = D == 3
            @test kinetic_energy(u, setup) isa Array{T}
            @test total_kinetic_energy(u, setup) isa T
            @test dissipation_from_strain(u, setup) isa Array{T}
            @test get_scale_numbers(u, setup) isa NamedTuple
        end
    end

    testops(IncompressibleNavierStokes.Dimension(2))
    testops(IncompressibleNavierStokes.Dimension(3))
end
