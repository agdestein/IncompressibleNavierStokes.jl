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
