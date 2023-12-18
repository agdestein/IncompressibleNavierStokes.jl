# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Decaying Homogeneous Isotropic Turbulence - 2D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

set_theme!(; GLMakie = (; scalefactor = 1.5))

# Output directory
output = "output/DecayingTurbulence2D"

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
Re = T(10_000)

# A 2D grid is a Cartesian product of two vectors
n = 8
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
pressure_solver = SpectralPressureSolver(setup);

u₀, p₀ = random_field(setup, T(0); pressure_solver);
u, p = u₀, p₀

using KernelAbstractions
using Zygote
using LinearAlgebra
using Random
(; Iu, Ip) = setup.grid

function finitediff(f, u::Tuple, I; h = sqrt(eps(eltype(u[1]))))
    u1 = copy.(u)
    CUDA.@allowscalar u1[1][I] -= h / 2
    r1 = f(u1)
    u2 = copy.(u)
    CUDA.@allowscalar u2[1][I] += h / 2
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
φ = IncompressibleNavierStokes.divergence!(zero(p), ur, setup)
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
    # dot(φ, φ)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
    # sum(abs2, getindex.(φ, Iu))
end

solver = SpectralPressureSolver(setup)
function f(u, setup)
    (; Ω, Iu) = setup.grid
    φ = u
    φ = IncompressibleNavierStokes.apply_bc_u(u, T(0), setup)
    φ = IncompressibleNavierStokes.momentum(φ, T(0), setup)
    φ = IncompressibleNavierStokes.apply_bc_u(φ, T(0), setup)
    φ = IncompressibleNavierStokes.project(solver, φ, setup)
    # dot(φ, φ)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
    # sum(abs2, getindex.(φ, Iu))
end

pressure_solver = SpectralPressureSolver(setup)
# method = 
function f(u, setup)
    (; Ω, Iu) = setup.grid
    method = RK44(; T)
    stepper = IncompressibleNavierStokes.create_stepper(
        method;
        setup,
        pressure_solver,
        u,
        p,
        t = T(0),
    )
    stepper = IncompressibleNavierStokes.timestep(method, stepper, T(1e-4))
    φ = stepper.u
    # dot(φ, φ)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
    # sum(abs2, getindex.(φ, Iu))
end

function f(u, setup)
    (; Iu) = setup.grid
    φ = IncompressibleNavierStokes.tupleadd(u, u)
    dot(getindex.(φ, Iu), getindex.(φ, Iu))
end

IncompressibleNavierStokes.tupleadd(u, u)

f(u, setup)

(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])
(g1 = gradient(u -> f(u, setup), u)[1]; KernelAbstractions.synchronize(get_backend(u[1])); g1[1][Iu[1]])

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
