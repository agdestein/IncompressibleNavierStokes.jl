if false
    include("src/SymmetryClosure.jl")
    using .SymmetryClosure
end

# using GLMakie
using CairoMakie
using IncompressibleNavierStokes
using SymmetryClosure
using CUDA
using Zygote
using Random
using LinearAlgebra
using NeuralClosure

lines(cumsum(randn(100)))

# Setup
n = 128
ax = range(0.0, 1.0, n + 1)
x = ax, ax
# x = tanh_grid(0.0, 1.0, n + 1), stretched_grid(-0.2, 1.2, n + 1)
setup = Setup(;
    x,
    Re = 1e4,
    backend = CUDABackend(),
    # boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
);
ustart = random_field(setup)
# ustart = vectorfield(setup) |> randn!

u = ustart

let
    B, V = tensorbasis(u, setup)
    # B, V = randn!(B), randn!(V)
    V = randn!(V)
    function f(u)
        Bi, Vi = tensorbasis(u, setup)
        # dot(Bi, B) + dot(Vi, V)
        # dot(getindex.(Bi, 1), getindex.(B, 1)) + dot(Vi, V)
        dot(Vi, V)
        # dot(Vi[:, :, 1], V[:, :, 1])
    end

    fd = map(eachindex(u)) do i
        h = 1e-2
        v1 = copy(u)
        v2 = copy(u)
        CUDA.@allowscalar v1[i] -= h / 2
        CUDA.@allowscalar v2[i] += h / 2
        (f(v2) - f(v1)) / h
    end |> x -> reshape(x, size(u))

    ad = Zygote.gradient(f, u)[1] |> Array

    # mask = @. abs(fd - ad) > 1e-3

    # i = 1
    # V[:, :, i] |> display
    # # (mask .* u)[:, :, i] |> display
    # (mask .* fd)[:, :, i] |> display
    # (mask .* ad)[:, :, i] |> display

    # fd .- ad |> display
    @show fd - ad .|> abs |> maximum
    # @show f(u)
    nothing
end

urot = NeuralClosure.rot2stag(u, 1)
B, V = tensorbasis(u, setup)
Brot, Vrot = tensorbasis(urot, setup)

iV = 2
# V[:, :, iV] |> Array |> heatmap
Vrot[:, :, iV] |> Array |> heatmap
rot2(V, 1)[:, :, iV] |> Array |> heatmap

Vrot - rot2(V, 1) .|> abs |> maximum

begin
    θ = randn(5, 3)
    # θ[:, 1] .= 0
    # θ[:, 3] .= 0
    # θ = zeros(5, 3)
    # θ[3, :] .= 1e-3 * randn()
    θ = θ |> CuArray
    s = tensorclosure(polynomial, u, θ, setup) |> u -> apply_bc_u(u, zero(eltype(u)), setup)
    srot =
        tensorclosure(polynomial, urot, θ, setup) |>
        u -> apply_bc_u(u, zero(eltype(u)), setup)
    rots = NeuralClosure.rot2stag(s, 1)
    (srot-rots)[2:end-1, 2:end-1, :] .|> abs |> maximum
end

i = 1
srot[2:end-1, 2:end-1, i] |> Array |> heatmap
rots[2:end-1, 2:end-1, i] |> Array |> heatmap
# s[2:end-1, 2:end-1, 2] |> Array |> heatmap
(srot-rots)[2:end-1, 2:end-1, i] |> Array |> heatmap
(srot-rots)[2:end-1, 2:end-1, :] .|> abs |> maximum
he = (srot-rots)[2:end-1, 2:end-1, :]
he[125:128, 1:10, 1]

x = randn(5, 5, 2)

typeof(B)
getindex.(B, 1)

B[:, :, 1]

tb, pb = SymmetryClosure.ChainRulesCore.rrule(tensorbasis, u, setup)

ubar = pb(tb)[2]

θ = 1e-5 * randn(5, 3)
θ = θ |> CuArray
s = tensorclosure(polynomial, ustart, θ, setup);

gradient(u -> sum(tensorclosure(polynomial, u, θ, setup)), ustart)[1];

using StaticArrays
tau = similar(setup.grid.x[1], SMatrix{2,2,Float64,4}, 10)
zero(tau)

s[2:end-1, 2:end-1, 1] |> heatmap
s[2:end-1, 2:end-1, 1] .|> abs |> heatmap
s[2:end-1, 2:end-1, 1] .|> abs .|> log |> heatmap

B |> length
B[2][5, 10]
getindex.(B[2], 2)

u[:, :, 1] |> heatmap
getindex.(B[3], 1, 1) |> heatmap
getindex.(B[3], 2, 1) |> heatmap
getindex.(B[3], 2, 2) |> heatmap
V[1][2:end-1, 2:end-1] |> heatmap
V[2][2:end-1, 2:end-1] |> heatmap

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (0.0, 1.0),
    processors = (log = timelogger(; nupdate = 10),),
);
(; u) = state

makeplot(u, fieldname, time) = save(
    "output/fieldplots/$fieldname$time.png",
    fieldplot((; u, t = T(0)); setup, fieldname, docolorbar = false, size = (500, 500)),
)

makeplot(ustart, :vorticity, 0)
makeplot(ustart, :S2, 0)
makeplot(ustart, :R2, 0)
makeplot(u, :vorticity, 1)
makeplot(u, :S2, 1)
makeplot(u, :R2, 1)
