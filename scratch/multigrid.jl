# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Train closure model
#
# Here, we consider a periodic box ``[0, 1]^2``. It is discretized with a
# uniform Cartesian grid with square cells.

using GLMakie
using IncompressibleNavierStokes
using JLD2
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Random
using Zygote

# Floating point precision
T = Float64

# Array type
ArrayType = Array
device = identity
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using LuxCUDA
using CUDA;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);
device = cu

# Parameters
get_params(nles) = (;
    D = 2,
    Re = T(10_000),
    lims = (T(0), T(1)),
    tburn = T(0.0),
    tsim = T(1.0),
    Δt = T(5e-4),
    nles,
    compression = 1024 ÷ nles,
    ArrayType,
    # ic_params = (; A = T(20_000_000), σ = T(5.0), s = T(3)),
)

nles = [8, 16, 32, 64, 128]

# Create LES data from DNS
data_train = [create_les_data(T; get_params(nles)..., nsim = 5) for nles in nles];
data_valid = [create_les_data(T; get_params(nles)..., nsim = 1) for nles in nles];
data_test = [create_les_data(T; get_params(nles)..., nsim = 1) for nles in nles];

# Inspect data
g = 5
j = 1
α = 1
data_train[g].u[j][1][α]
o = Observable(data_train[g].u[j][1][α])
heatmap(o)
for i = 1:1:length(data_train[g].u[j])
    o[] = data_train[g].u[j][i][α]
    sleep(0.001)
end

# # Save filtered DNS data
# jldsave("output/forced/data.jld2"; data_train, data_valid, data_test)

# # Load previous LES data
# data_train, data_valid, data_test = load("output/forced/data.jld2", "data_train", "data_valid", "data_test")

# Build LES setup and assemble operators
x = ntuple(α -> LinRange(params.lims..., params.nles + 1), params.D)
setup = Setup(x...; params.Re, ArrayType);

# Uniform periodic grid
pressure_solver = SpectralPressureSolver(setup);

closure, θ₀ = cnn(
    setup,

    # Radius
    [2, 2, 2, 2],

    # Channels
    [5, 5, 5, params.D],

    # Activations
    [leakyrelu, leakyrelu, leakyrelu, leakyrelu, identity],

    # Bias
    [true, true, true, true, false];
);
closure.NN

# closure, θ₀ = fno(
#     setup,
#
#     # Cut-off wavenumbers
#     [8, 8, 8, 8],
#
#     # Channel sizes
#     [8, 8, 8, 8],
#
#     # Fourier layer activations
#     [gelu, gelu, gelu, identity],
#
#     # Dense activation
#     gelu,
# );
# closure.NN

# Create input/output arrays
io_train = create_io_arrays(data_train, setup);
io_valid = create_io_arrays(data_valid, setup);
io_test = create_io_arrays(data_test, setup);

size(io_train[1])

# Prepare training
θ = T(1.0e-1) * device(θ₀);
# θ = device(θ₀);
opt = Optimisers.setup(Adam(T(1.0e-3)), θ);
callbackstate = Point2f[];
randloss = create_randloss(mean_squared_error, closure, io_train...; nuse = 50, device);

# Warm-up
randloss(θ)
@time randloss(θ);
first(gradient(randloss, θ));
@time first(gradient(randloss, θ));
GC.gc()
CUDA.reclaim()

# Training
# Note: The states `opt`, `θ`, and `callbackstate`
# will not be overwritten until training is finished.
# This allows for cancelling with "Control-C" should errors explode.
(; opt, θ, callbackstate) = train(
    randloss,
    opt,
    θ;
    niter = 50,
    ncallback = 10,
    callbackstate,
    callback = create_callback(closure, device(io_valid)...; state = callbackstate),
);
GC.gc()
CUDA.reclaim()

Array(θ)

# # Save trained parameters
# jldsave("output/forced/theta_cnn.jld2"; theta = Array(θ))
# jldsave("output/forced/theta_fno.jld2"; theta = Array(θ))

# # Load trained parameters
# θθ = load("output/theta_cnn.jld2")
# θθ = load("output/theta_fno.jld2")
# copyto!(θ, θθ["theta"])
