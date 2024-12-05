if false
    include("src/SymmetryClosure.jl")
    using .SymmetryClosure
end

using CairoMakie
using IncompressibleNavierStokes
using SymmetryClosure
using CUDA
using Zygote

lines(cumsum(randn(100)))

# Setup
n = 32
ax = range(0.0, 1.0, n + 1)
setup = Setup(; x = (ax, ax), Re = 1e4, backend = CUDABackend());
ustart = random_field(setup, 0.0)

u = ustart

B, V = IncompressibleNavierStokes.tensorbasis(u, setup)

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
