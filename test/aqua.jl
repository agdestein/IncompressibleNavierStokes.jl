@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(IncompressibleNavierStokes)
end
