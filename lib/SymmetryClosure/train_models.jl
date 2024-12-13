# LSP hack (to get "go to definition" etc. working)    #src
if false                                               #src
    include("src/SymmetryClosure.jl")                  #src
    include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    using .NeuralClosure                               #src
    using .IncompressibleNavierStokes                  #src
end                                                    #src

########################################################################## #src

# # Train models
#
# This script is used to train models.

@info "# Train models"

########################################################################## #src

# ## Packages

@info "Loading packages"
flush(stdout)

using Accessors
using CairoMakie
using IncompressibleNavierStokes
using Lux
using LuxCUDA
using Optimisers
using ParameterSchedulers
using NeuralClosure
using Random
using SymmetryClosure
using Zygote

########################################################################## #src

# ## Setup

# Get SLURM specific variables (if any)
(; jobid, taskid) = slurm_vars()

# Log info about current run
time_info()

# Hardware selection
(; backend, device, clean) = hardware()

# Test case
(; params, outdir, plotdir, seed_dns, ntrajectory) = testcase(backend)

# DNS seeds
dns_seeds = splitseed(seed_dns, ntrajectory)
dns_seeds_test = dns_seeds[1:1]
dns_seeds_valid = dns_seeds[2:2]
dns_seeds_train = dns_seeds[3:end]

########################################################################## #src

# ## Model definitions

setups = map(nles -> getsetup(; params, nles), params.nles)

tensorcoeffs, θ_start = polynomial, zeros(5, 3)
tensorcoeffs, θ_start = create_cnn(;
    setup = setups[1],
    radii = [2, 2, 2],
    channels = [12, 12, 3],
    activations = [tanh, tanh, identity],
    use_bias = [true, true, false],
    rng = Xoshiro(123),
);
tensorcoeffs.m.chain
closure_models = map(s -> tensorclosure(tensorcoeffs, s), setups)

# Test run
let
    data_test = map(
        s -> namedtupleload(getdatafile(outdir, params.nles[1], params.filters[1], s)),
        dns_seeds_test,
    )
    @show size(data_test[1].u)
    sample = data_test[1].u[:, :, :, 1] |> device
    θ = θ_start |> device # |> randn! |> x -> x ./ 1e6
    closure_models[1](sample, θ)
    gradient(θ -> sum(closure_models[1](sample, θ)), θ)[1]
end

# Train
let
    nepoch = 10
    T = typeof(params.Re)
    trainpost(;
        params,
        outdir,
        plotdir,
        taskid,
        postseed = 123,
        dns_seeds_train,
        dns_seeds_valid,
        nsubstep = 5,
        nunroll = 10,
        ntrajectory = 5,
        closure_models,
        θ_start,
        opt = Adam(T(1e-4)),
        λ = T(5e-8),
        scheduler = CosAnneal(; l0 = T(1e-6), l1 = T(1e-3), period = nepoch),
        nunroll_valid = 50,
        nupdate_callback = 10,
        displayref = false,
        displayupdates = true,
        loadcheckpoint = false,
        nepoch,
        niter = 100,
    )
end
