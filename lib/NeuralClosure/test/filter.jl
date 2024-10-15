@testitem "Filters" begin
    using IncompressibleNavierStokes
    D = 2
    n = 16, 32, 64
    setups = map(n -> Setup(; x = ntuple(d -> range(0.0, 1.0, n + 1), D), Re = 1e3), n)
    u = vectorfield.(setups)
    val = 3.83
    fill!.(u[end], val)
    compression = div.(n[end], n)
    filters = FaceAverage(), VolumeAverage()
    for Φ in filters, (setup, comp) in zip(setups, compression)
        (; Iu) = setup.grid
        v = Φ(u[end], setup, comp)
        for i = 1:D
            @test all(≈(val), v[i][Iu[i]])
        end
    end
end
