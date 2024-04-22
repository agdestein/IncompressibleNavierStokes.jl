using CairoMakie
using IncompressibleNavierStokes

# For running on GPU
using CUDA;
CUDA.allowscalar(false);
ArrayType = CuArray;
T = Float32

# Setup
Re = T(10_000)
n = 1024
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = Setup(x...; Re, ArrayType);
psolver = SpectralPressureSolver(setup);
u₀ = random_field(setup, T(0); psolver);

u = u₀

B, V = IncompressibleNavierStokes.tensorbasis(u, setup)
B |> length
CUDA.@allowscalar B[2][5, 10]
getindex.(B[2], 2)

heatmap(Array(u[1]))
u[1] |> Array |> heatmap
getindex.(B[3], 1, 1) |> Array |> heatmap
getindex.(B[3], 2, 1) |> Array |> heatmap
getindex.(B[3], 2, 2) |> Array |> heatmap
V[1] |> Array |> heatmap
V[2] |> Array |> heatmap
V[2] |> Array |> contourf

# Solve unsteady problem
state, outputs = solve_unsteady(
    setup,
    u₀,
    (T(0), T(1));
    Δt = T(1e-4),
    psolver,
    processors = (
        # rtp = realtimeplotter(; setup, nupdate = 1),
        log = timelogger(; nupdate = 10),
    ),
);
(; u) = state

makeplot(u, fieldname, time) = save(
    "output/fieldplots/$fieldname$time.png",
    fieldplot((; u, t = T(0)); setup, fieldname, docolorbar = false, size = (500, 500)),
)

makeplot(u₀, :vorticity, 0)
makeplot(u₀, :S2, 0)
makeplot(u₀, :R2, 0)
makeplot(u, :vorticity, 1)
makeplot(u, :S2, 1)
makeplot(u, :R2, 1)
