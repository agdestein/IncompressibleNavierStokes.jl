using IncompressibleNavierStokes
using NeuralClosure
using NNlib
using Lux
using Random

T = Float32
rng = Xoshiro()

gconvs = (
    GroupConv2D((3, 3), 1 => 5; init_bias = glorot_normal, activation = tanh),
    GroupConv2D((3, 3), 5 => 5; init_bias = glorot_normal, activation = tanh),
    GroupConv2D((3, 3), 5 => 5; init_bias = glorot_normal, activation = tanh),
    GroupConv2D((3, 3), 5 => 1; use_bias = false),
)

params = map(c -> Lux.setup(rng, c)[1], gconvs);

function gcnn(u, params, layers)
    u = (u[1], u[2], -u[1], -u[2])
    for i in 1:length(layers)
        u = layers[i](u, params[i], (;))[1]
    end
    u = (u[1] - u[3], u[2] - u[4])
end

n = 100
ux = randn(T, n - 2, n - 2, 1, 1)
uy = randn(T, n - 2, n - 2, 1, 1)
u = (ux, uy)
u = pad_circular.(u, 1)
ru = rot2(u, 2)

c = gcnn(u, params, gconvs)
cr = gcnn(ru, params, gconvs)
rc = rot2(c, 2)

cr[1]
rc[1]
cr[2]
rc[2]

cr[1] - rc[1]
cr[2] - rc[2]
