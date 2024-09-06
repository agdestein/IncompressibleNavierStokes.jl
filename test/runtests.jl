# Add environment for package being tested via
# load path for live testing, since the test Project.toml
# is not allowed to depend on the tested package for Pkg.test()
# to work correctly
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Aqua
using CairoMakie
using ChainRulesCore
using ChainRulesTestUtils
using IncompressibleNavierStokes
using IncompressibleNavierStokes: convectiondiffusion!
using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Test

@testset "IncompressibleNavierStokes" begin
    include("grid.jl")
    include("psolvers.jl")
    include("operators.jl")
    include("chainrules.jl")
    # include("timesteppers.jl")
    include("postprocess.jl")
    include("aqua.jl")
end
