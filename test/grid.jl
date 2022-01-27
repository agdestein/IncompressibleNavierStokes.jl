@testset "Grid" begin
    a = 1.2
    b = 4.5
    N = 76

    @testset "Cosine grid" begin
        x = cosine_grid(a, b, N)
        @test length(x) == N + 1
        @test x[1] ≈ a
        @test x[end] ≈ b
        @test all(diff(x) .> 0)
    end

    @testset "Stretched grid" begin
        @test_throws ErrorException stretched_grid(a, b, N, -0.1)
        @test_throws ErrorException stretched_grid(a, b, N, 0)

        x = stretched_grid(a, b, N, 0.95)
        @test length(x) == N + 1
        @test x[1] ≈ a
        @test x[end] ≈ b
        @test all(diff(x) .> 0)
        @test all(diff(diff(x)) .< 0)

        x = stretched_grid(a, b, N, 1)
        @test length(x) == N + 1
        @test x[1] ≈ a
        @test x[end] ≈ b
        @test all(diff(x) .≈ (b - a) / N)

        x = stretched_grid(a, b, N, 1.05)
        @test length(x) == N + 1
        @test x[1] ≈ a
        @test x[end] ≈ b
        @test all(diff(x) .> 0)
        @test all(diff(diff(x)) .> 0)
    end
end
