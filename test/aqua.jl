@testset "Aqua" begin
    @info "Testing code with Aqua"
    Aqua.test_all(IncompressibleNavierStokes; ambiguities = false)
end
