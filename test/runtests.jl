push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Aqua
using CairoMakie
using ChainRulesCore
using ChainRulesTestUtils
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    apply_bc_p,
    apply_bc_u,
    bodyforce,
    convection,
    convection!,
    convectiondiffusion!,
    diffusion,
    diffusion!,
    divergence,
    eig2field,
    kinetic_energy,
    interpolate_u_p,
    interpolate_ω_p,
    laplacian,
    laplacian_mat,
    momentum,
    poisson,
    pressuregradient,
    smagorinsky_closure,
    tensorbasis,
    total_kinetic_energy,
    vorticity,
    Dfield,
    Qfield

using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Test

@testset "IncompressibleNavierStokes" begin
    include("grid.jl")
    include("pressure_solvers.jl")
    include("operators.jl")
    include("chainrules.jl")
    # include("models.jl")
    # include("solvers.jl")
    # include("simulation2D.jl")
    # include("simulation3D.jl")
    # include("postprocess2D.jl")
    # include("postprocess3D.jl")

    @testset "Aqua" begin
        @info "Testing code with Aqua"
        Aqua.test_all(IncompressibleNavierStokes; ambiguities = false)
    end
end
