using IncompressibleNavierStokes
using GLMakie
using CairoMakie
using NeuralClosure
using SymmetryClosure
using LinearAlgebra

T = Float64
ArrayType = Array
n = 64
setup =
    Setup(LinRange(T(0), T(1), n + 1), LinRange(T(0), T(1), n + 1); Re = T(2000), ArrayType);
psolver = psolver_spectral(setup);
ustart = random_field(setup, T(0); psolver);

state, _ = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(0.5)),
    processors = (
        # rtp = realtimeplotter(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

ustart_rot = rot2stag(ustart, 1);

state_rot, _ = solve_unsteady(;
    setup,
    ustart = ustart_rot,
    tlims = (T(0), T(0.5)),
    processors = (
        # rtp = realtimeplotter(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

let
    u, v = rot2(state.u, 1)
    u, v = state_rot.u
    x = setup.grid.xp[1][2:end-1]
    y = setup.grid.xp[1][2:end-1]
    ux = state.u[1][setup.grid.Iu[1]]
    uy = state.u[2][setup.grid.Iu[2]]
    n = sqrt.(ux .^ 2 .+ uy .^ 2)
    arrows(
        x,
        y,
        ux,
        uy;
        # arrowsize = vec(n),
        arrowsize = 10,
        arrowcolor = vec(n),
        linecolor = vec(n),
        lengthscale = 0.04,
        figure = (; size = (500, 500)),
    )
end

let
    F = IncompressibleNavierStokes.momentum(ustart, nothing, T(0), setup)
    FR = IncompressibleNavierStokes.momentum(ustart_rot, nothing, T(0), setup)
    IncompressibleNavierStokes.apply_bc_u!(F, T(0), setup)
    RF = rot2stag(F, 1)
    IncompressibleNavierStokes.apply_bc_u!(RF, T(0), setup)
    i = 2
    # a = RF[i]
    # b = FR[i]
    a = RF[i][setup.grid.Iu[i]]
    b = FR[i][setup.grid.Iu[i]]
    norm(a - b) / norm(b)
    a - b
end

using Random
using NeuralClosure

closure, θ₀ = gcnn(;
    setup,
    radii = [2, 2, 2, 2, 2],
    channels = [6, 6, 6, 6, 1],
    activations = [tanh, tanh, tanh, tanh, identity],
    use_bias = [true, true, true, true, false],
    rng = Xoshiro(),
);

closure.chain

u = ustart
v = (u[1][2:end-1, 2:end-1], u[2][2:end-1, 2:end-1])
rv = rot2stag(v, 1)
w = cat(stack(v); dims = 4)
rw = cat(stack(rv); dims = 4)
c = closure(w, θ₀)
cr = closure(rw, θ₀)
rc = rot2stag((c[:, :, 1, 1], c[:, :, 2, 1]), 1)

cr[:, :, 1, 1] |> heatmap
rc[1] |> heatmap

m = wrappedclosure(closure, setup)

rm = rot2stag(m(state.u, θ₀), 1)
mr = m(rot2stag(state.u, 1), θ₀)

rm[1]
mr[1]

ran = 0.4
heatmap(rm[2]; colorrange = (-ran, ran))
heatmap(mr[2]; colorrange = (-ran, ran))

a = fill(10, 5) .+ (1:5)'
b = (1:5) .* (1:5)'
x = rot2((a, b), 1)
x[1]
x[2]

x = LinRange(0, 2pi, 20)
y = LinRange(0, 3pi, 20)
u = @. sin(x) * cos(y') * (x < pi) * (y' > pi)
v = @. -cos(x) * sin(y') * (x < pi) * (y' > pi)
strength = sqrt.(u .^ 2 .+ v .^ 2)

v

arrows(
    x,
    y,
    u,
    v;
    arrowsize = 10,
    lengthscale = 0.3,
    arrowcolor = vec(strength),
    linecolor = vec(strength),
    figure = (; size = (600, 600)),
)

g = 3
arrows(
    y,
    x,
    # x, y,
    rot2((u, v), g)...;
    arrowsize = 10,
    lengthscale = 0.3,
    arrowcolor = vec(rot2(strength, g)),
    linecolor = vec(rot2(strength, g)),
    figure = (; size = (600, 600)),
)
