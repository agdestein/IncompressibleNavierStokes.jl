using IncompressibleNavierStokes
using NeuralClosure
using NNlib
using Lux
using Random

T = Float64
rng = Xoshiro()

init_bias = glorot_normal
# init_bias = zeros32
gcnn = Chain(
    GroupConv2D((3, 3), 1 => 5, tanh; init_bias, islifting = true),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 1; use_bias = false, isprojecting = true),
)

params, state = Lux.setup(rng, gcnn) |> f64

n = 100
ux = randn(T, n, n, 1, 1)
uy = randn(T, n, n, 1, 1)
u = (ux, uy)
ru = rot2(u, 1)
u = cat(u...; dims = 3)
ru = cat(ru...; dims = 3)

c, _ = gcnn(u, params, state)
cr, _ = gcnn(ru, params, state)
c = (c[:, :, 1, 1], c[:, :, 2, 1])
cr = (cr[:, :, 1, 1], cr[:, :, 2, 1])
rc = rot2(c, 1)

using LinearAlgebra
norm(cr[1] - rc[1]) / norm(rc[1])
norm(cr[2] - rc[2]) / norm(rc[2])
