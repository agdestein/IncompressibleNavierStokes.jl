using IncompressibleNavierStokes
using NeuralClosure
using Lux
using Random
using Zygote
using ComponentArrays

T = Float32
rng = Xoshiro()

init_bias = glorot_normal
c = Chain(
    GroupConv2D((3, 3), 1 => 5, tanh; init_bias, islifting = true),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 5, tanh; init_bias),
    GroupConv2D((3, 3), 5 => 1; use_bias = false, isprojecting = true),
)

# c = GroupConv2D((3, 3), 5 => 5, tanh; init_bias)

params, state = Lux.setup(rng, c)

Lux.parameterlength(c)

n = 100
ux = randn(T, n, n, 1, 1)
uy = randn(T, n, n, 1, 1)
u = (ux, uy)
u = cat(u...; dims = 3)

c(u, params, state)

sum(abs2, c(u, params, state)[1])

gradient(u -> sum(abs2, c(u, params, state)[1]), u)
gradient(p -> sum(abs2, c(u, p, state)[1]), params)
