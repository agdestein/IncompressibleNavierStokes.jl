# LSP indexing solution
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using LinearAlgebra
using Test

@testset "IncompressibleNavierStokes" begin
    include("simulation2D.jl")
    include("simulation3D.jl")
    include("models.jl")
end
