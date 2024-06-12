using IncompressibleNavierStokes
using NeuralClosure
using Lux
using Random
rng = Xoshiro()

conv = Conv((5,), 2 => 8)
params, state = Lux.setup(rng, conv)
params
