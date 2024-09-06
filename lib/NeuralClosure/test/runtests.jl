using Aqua
using IncompressibleNavierStokes
using NeuralClosure
using Test

@testset "NeuralClosure" begin
    @testset "Example run" begin
        include("examplerun.jl")
    end
    @testset "Aqua" begin
        @info "Testing code with Aqua"
        Aqua.test_all(NeuralClosure; ambiguities = false)
    end
end
