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

# # Data generation
#
# This script is used to generate filtered DNS data.

@info "# DNS data generation"

########################################################################## #src

# ## Setup

using SymmetryClosure
using Random
using NeuralClosure
using JLD2

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

########################################################################## #src

# ## Create data

for (iseed, seed) in enumerate(dns_seeds)
    if isnothing(taskid) || iseed == taskid
        @info "Creating DNS trajectory for seed $(repr(seed))"
    else
        # Each task does one initial condition
        @info "Skipping seed $(repr(seed)) for task $taskid"
        continue
    end
    filenames = map(Iterators.product(params.nles, params.filters)) do (nles, Φ)
        f = getdatafile(outdir, nles, Φ, seed)
        datadir = dirname(f)
        ispath(datadir) || mkpath(datadir)
        f
    end
    data = create_les_data(; params..., backend, rng = Xoshiro(seed), filenames)
    @info(
        "Trajectory info:",
        data[1].comptime / 60,
        length(data[1].t),
        Base.summarysize(data) * 1e-9,
    )
end

########################################################################## #src

# Computational time

docomp = true
docomp && let
    comptime, datasize = 0.0, 0.0
    for seed in dns_seeds
        comptime += load(
            getdatafile(outdir, params.nles[1], params.filters[1], seed),
            "comptime",
        )
    end
    for seed in dns_seeds, nles in params.nles, Φ in params.filters
        data = namedtupleload(getdatafile(outdir, nles, Φ, seed))
        datasize += Base.summarysize(data)
    end
    @info "Data" comptime
    @info "Data" comptime / 60 datasize * 1e-9
    clean()
end
