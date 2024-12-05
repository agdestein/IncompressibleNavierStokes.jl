if false
    include("src/SymmetryClosure.jl")
    using .SymmetryClosure
end

using CairoMakie
using IncompressibleNavierStokes
using SymmetryClosure
using CUDA
using Zygote
using Random
using LinearAlgebra

lines(cumsum(randn(100)))

# Setup
n = 8
# ax = range(0.0, 1.0, n + 1)
# x = ax, ax
x = tanh_grid(0.0, 1.0, n + 1), stretched_grid(-0.2, 1.2, n + 1)
setup = Setup(;
    x,
    Re = 1e4,
    backend = CUDABackend(),
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
);
ustart = vectorfield(setup) |> randn!

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

B, V = tensorbasis(u, setup)

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
