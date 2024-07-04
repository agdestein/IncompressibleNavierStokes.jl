# Add environment for package being tested via
# load path for live testing, since the test Project.toml
# is not allowed to depend on the tested package for Pkg.test()
# to work correctly
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Aqua
using IncompressibleNavierStokes
using NeuralClosure
using Test

@testset "NeuralClosure" begin
    @testset "Aqua" begin
        @info "Testing code with Aqua"
        Aqua.test_all(NeuralClosure; ambiguities = false)
    end
end
