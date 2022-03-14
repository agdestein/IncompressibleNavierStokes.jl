# LSP indexing solution
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using LinearAlgebra
using Statistics
using Test
using Aqua

@testset "IncompressibleNavierStokes" begin
    include("grid.jl")
    include("pressure_solvers.jl")
    include("models.jl")
    include("solvers.jl")
    include("simulation2D.jl")
    include("simulation3D.jl")

    @testset "Aqua" begin
        Aqua.test_all(
            IncompressibleNavierStokes;
            ambiguities = false,
            project_toml_formatting = false, # https://github.com/JuliaTesting/Aqua.jl/issues/72
        )
    end
end
