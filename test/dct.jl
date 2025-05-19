@testitem "Permutation indices for DCT" begin
    using IncompressibleNavierStokes: get_perminds

    # Test that permutation works
    N = 10
    perm, perminv = get_perminds(N)
    u = randn(N)
    @test u[perm][perminv] == u[perminv][perm] == u

    # It should only accept even N for now
    @test_throws AssertionError get_perminds(11)
end

@testitem "Discrete Cosine Transform" begin
    using IncompressibleNavierStokes: manual_dct_stuff, manual_dct!, manual_idct!
    using FFTW

    # 1D version
    u = randn(30)
    stuff = manual_dct_stuff(u)
    @test manual_dct!(copy(u), 1, stuff) ≈ dct(u, 1)
    @test manual_idct!(copy(u), 1, stuff) ≈ idct(u, 1)

    # It should only allow reals for now
    @test_throws AssertionError manual_dct!(u .+ im, 1, stuff)

    # 2D version
    u = randn(20, 26)
    stuff = manual_dct_stuff(u)
    @test manual_dct!(copy(u), 1, stuff) ≈ dct(u, 1)
    @test manual_dct!(copy(u), 2, stuff) ≈ dct(u, 2)
    @test manual_idct!(copy(u), 1, stuff) ≈ idct(u, 1)
    @test manual_idct!(copy(u), 2, stuff) ≈ idct(u, 2)

    # 3D version
    u = randn(10, 16, 28)
    stuff = manual_dct_stuff(u)
    @test manual_dct!(copy(u), 1, stuff) ≈ dct(u, 1)
    @test manual_dct!(copy(u), 2, stuff) ≈ dct(u, 2)
    @test manual_dct!(copy(u), 3, stuff) ≈ dct(u, 3)
    @test manual_idct!(copy(u), 1, stuff) ≈ idct(u, 1)
    @test manual_idct!(copy(u), 2, stuff) ≈ idct(u, 2)
    @test manual_idct!(copy(u), 3, stuff) ≈ idct(u, 3)
end
