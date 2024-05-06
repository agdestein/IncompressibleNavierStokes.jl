using Aqua
using IncompressibleNavierStokes
using NeuralClosure

@testset "NeuralClosure" begin
    @testset "Aqua" begin
        @info "Testing code with Aqua"
        Aqua.test_all(NeuralClosure; ambiguities = false)
    end
end
