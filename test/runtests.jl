push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Aqua
using CairoMakie
using ChainRulesCore
using ChainRulesTestUtils
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    apply_bc_p,
    apply_bc_u,
    apply_bc_temp,
    bodyforce,
    convection_diffusion_temp,
    convection,
    convection!,
    convectiondiffusion!,
    diffusion,
    diffusion!,
    dissipation,
    dissipation_from_strain,
    divergence,
    eig2field,
    get_scale_numbers,
    gravity,
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
    include("psolvers.jl")
    include("operators.jl")
    include("chainrules.jl")
    # include("timesteppers.jl")
    # include("simulation.jl")
    # include("postprocess.jl")
    include("aqua.jl")
end
