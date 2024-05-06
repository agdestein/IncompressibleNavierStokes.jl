# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using KernelAbstractions
using Zygote
using LinearAlgebra
using Random

set_theme!(; GLMakie = (; scalefactor = 1.5))

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using CUDA;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);

# Viscosity model
Re = T(100)

# A 2D grid is a Cartesian product of two vectors
# n = 8
n = 16
# n = 32
# n = 128
# n = 1024
# n = 2056
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
# plotgrid(x...)

# Build setup and assemble operators
setup = Setup(x...; Re, ArrayType);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
psolver = SpectralPressureSolver(setup);

u₀ = random_field(setup, T(0); psolver);
u = u₀

(; Iu, Ip) = setup.grid

function finitediff(f, u::Tuple, α, I; h = sqrt(eps(eltype(u[1]))))
    u1 = copy.(u)
    CUDA.@allowscalar u1[α][I] -= h / 2
    r1 = f(u1)
    u2 = copy.(u)
    CUDA.@allowscalar u2[α][I] += h / 2
    r2 = f(u2)
    (r2 - r1) / h
end

function finitediff(f, p, I; h = sqrt(eps(eltype(p))))
    p1 = copy(p)
    CUDA.@allowscalar p1[I] -= h / 2
    r1 = f(p1)
    p2 = copy(p)
    CUDA.@allowscalar p2[I] += h / 2
    r2 = f(p2)
    (r2 - r1) / h
end

IncompressibleNavierStokes.diffusion(u, setup)[1]
IncompressibleNavierStokes.diffusion_adjoint!(zero.(u), u, setup)[1]

gradient(u -> sum(IncompressibleNavierStokes.diffusion(u, setup)[1]), u)[1][1]

# Divergence
ur = randn!.(similar.(u))
φ = IncompressibleNavierStokes.divergence(ur, setup)
φbar = randn!(similar(φ))
dot(φbar, φ)
dot(IncompressibleNavierStokes.divergence_adjoint!(zero.(ur), φbar, setup), ur)

# Diffusion
φ = IncompressibleNavierStokes.diffusion(u, setup)
φbar = randn!.(similar.(φ))
ubar = IncompressibleNavierStokes.diffusion_adjoint!(zero.(u), φbar, setup)
dot(φbar, φ)
dot(ubar, u)

# Convection
ur = randn!.(similar.(u))
# ur = zero.(u)
# for α = 1:length(ur)
#     ur[α][Iu[α]] .= randn.()
# end
# ur = u
φ = IncompressibleNavierStokes.convection!(zero.(ur), ur, setup)
φbar = randn!.(similar.(φ))
ubar = IncompressibleNavierStokes.convection_adjoint!(zero.(ur), φbar, ur, setup)
dot(φbar, φ)
dot(ubar, ur) / 2

function f(u, setup)
    (; Iu) = setup.grid
    u = IncompressibleNavierStokes.apply_bc_u(u, T(0), setup)
    φ = IncompressibleNavierStokes.momentum(u, T(0), setup)
    # φ = IncompressibleNavierStokes.diffusion(u, setup)
    # φ = IncompressibleNavierStokes.convection(u, setup)
    # φ = u
    # dot(φ, φ)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
    # sum(abs2, getindex.(φ, Iu))
    # u[1][1]
end

I = CartesianIndex(1, 2)
α = 1
CUDA.@allowscalar gradient(u -> f(u, setup), u)[1][α][I]
finitediff(u -> f(u, setup), u, α, I)

function f(u, setup)
    (; Ω, Iu) = setup.grid
    φ = u
    φ = IncompressibleNavierStokes.apply_bc_u(u, T(0), setup)
    φ = IncompressibleNavierStokes.momentum(φ, T(0), setup)
    φ = IncompressibleNavierStokes.apply_bc_u(φ, T(0), setup)
    φ = IncompressibleNavierStokes.project(φ, setup; psolver)
    # dot(φ, φ)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
    # sum(abs2, getindex.(φ, Iu))
end

I = CartesianIndex(8, 2)
α = 1
CUDA.@allowscalar gradient(u -> f(u, setup), u)[1][α][I]
finitediff(u -> f(u, setup), u, α, I)

function f(u, setup)
    (; Ω, Iu) = setup.grid
    method = RK44(; T)
    stepper = IncompressibleNavierStokes.create_stepper(method; setup, psolver, u, t = T(0))
    stepper = IncompressibleNavierStokes.timestep(method, stepper, T(1e-4))
    φ = stepper.u
    # dot(φ, φ)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
    # sum(abs2, getindex.(φ, Iu))
end

I = CartesianIndex(5, 6)
α = 1
CUDA.@allowscalar gradient(u -> f(u, setup), u)[1][α][I]
finitediff(u -> f(u, setup), u, α, I)

function f(u, setup)
    (; Iu) = setup.grid
    φ = IncompressibleNavierStokes.tupleadd(u, u)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
end

IncompressibleNavierStokes.tupleadd(u, u)

f(u, setup)

I = CartesianIndex(2, 2)
CUDA.@allowscalar gradient(u -> f(u, setup), u)[1][1][I]
finitediff(u -> f(u, setup), u, I)

function fp(p, setup)
    (; Ip) = setup.grid
    p = IncompressibleNavierStokes.apply_bc_p(p, T(0), setup)
    φ = IncompressibleNavierStokes.pressuregradient(p, setup)
    dot(φ, φ)
    # dot(getindex.(φ, Iu), getindex.(φ, Iu))
    # sum(abs2, getindex.(φ, Iu))
end

I = CartesianIndex(2, 2)
CUDA.@allowscalar gradient(p -> fp(p, setup), p)[1][I]
finitediff(p -> fp(p, setup), p, I)

φ[1]
φbar[1]
ur[1]
ubar[1]
