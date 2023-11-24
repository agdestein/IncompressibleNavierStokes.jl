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
    Re = T(6_000),
    lims = (T(0), T(1)),
    tburn = T(0.05),
    tsim = T(0.05),
    Δt = T(1e-4),
    nles,
    compression = 2048 ÷ nles,
    ArrayType,
    # ic_params = (; A = T(20_000_000), σ = T(5.0), s = T(3)),
    # ic_params = (; A = T(10)),
)

# Create LES data from DNS
data_train = [create_les_data(T; get_params(nles)..., nsim = 5) for nles in [32, 64, 128]];
data_valid = [create_les_data(T; get_params(nles)..., nsim = 1) for nles in [128]];
ntest = [8, 16, 32, 64, 128, 256, 512]
data_test = [create_les_data(T; get_params(nles)..., nsim = 1) for nles in ntest];

# Inspect data
g = 4
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

relerr_track(uref, setup) = processor() do state
    (; dimension, x, Ip) = setup.grid
    D = dimension()
    T = eltype(x[1])
    e = Ref(T(0))
    on(state) do (; u, n)
        a, b = T(0), T(0)
        for α = 1:D
            # @show size(uref[n + 1])
            a += sum(abs2, u[α][Ip] - uref[n+1][α][Ip])
            b += sum(abs2, uref[n+1][α][Ip])
        end
        e[] += sqrt(a) / sqrt(b) / (length(uref) - 1)
    end
    e
end

e_nm = zeros(T, length(ntest))
for (i, n) in enumerate(ntest)
    params = get_params(n)
    x = ntuple(α -> LinRange(params.lims..., params.nles + 1), params.D)
    setup = Setup(x...; params.Re, ArrayType)
    pressure_solver = SpectralPressureSolver(setup)
    u = device.(data_test[i].u[1])
    u₀ = device(data_test[i].u[1][1])
    p₀ = pressure_additional_solve(pressure_solver, u₀, T(0), setup)
    tlims = (T(0), params.tsim)
    (; Δt) = data_test[i]
    processors = (; relerr = relerr_track(u, setup))
    _, _, o = solve_unsteady(setup, u₀, p₀, tlims; Δt, pressure_solver, processors)
    e_nm[i] = o.relerr[]
    _, _, o = solve_unsteady(setup, u₀, p₀, tlims; Δt, pressure_solver, processors)
    e_cnn[i] = o.relerr[]
end
e_nm
e_cnn = ones(T, length(ntest))
e_fno_share = ones(T, length(ntest))
e_fno_spec = ones(T, length(ntest))

using CairoMakie
CairoMakie.activate!()

# Plot convergence
with_theme(;
# linewidth = 5,
# markersize = 20,
# fontsize = 20,
) do
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xticks = ntest,
        xlabel = "n",
        title = "Relative error (DNS: n = 2048)",
    )
    scatterlines!(ntest, e_nm; label = "No closure")
    scatterlines!(ntest, e_cnn; label = "CNN")
    scatterlines!(ntest, e_fno_spec; label = "FNO (retrained)")
    scatterlines!(ntest, e_fno_share; label = "FNO (shared parameters)")
    lines!(collect(extrema(ntest)), n -> 100n^-2.0; linestyle = :dash, label = "n^-2")
    axislegend(; position = :lb)
    fig
end

save("convergence.pdf", current_figure())

closure, θ₀ = cnn(
    setup,
    radii = [2, 2, 2, 2],
    channels = [5, 5, 5, params.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, false],
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
