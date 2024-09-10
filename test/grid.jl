@testitem "Cosine grid" begin
    a, b, N = 1.2, 4.5, 76
    x = cosine_grid(a, b, N)
    @test length(x) == N + 1
    @test x[1] ≈ a
    @test x[end] ≈ b
    @test all(diff(x) .> 0)
end

@testitem "Stretched grid" begin
    a, b, N = 1.2, 4.5, 76
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

@testitem "Tanh grid" begin
    a, b, N = 1.2, 4.5, 76
    x = tanh_grid(a, b, N, 2)
    @test length(x) == N + 1
    @test x[1] ≈ a
    @test x[end] ≈ b
    @test all(diff(x) .> 0)
end
