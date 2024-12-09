# LSP hack                                             #src
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

@info "Loading packages"
flush(stdout)

using Adapt
using CUDA
using Dates
using JLD2
using NeuralClosure
using Random
using SymmetryClosure

########################################################################## #src

# SLURM specific variables
jobid = haskey(ENV, "SLURM_JOB_ID") ? parse(Int, ENV["SLURM_JOB_ID"]) : nothing
taskid =
    haskey(ENV, "SLURM_ARRAY_TASK_ID") ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : nothing

isnothing(jobid) || @info "Running on SLURM (jobid = $jobid)"
isnothing(taskid) || @info "Task id = $taskid)"

########################################################################## #src

@info "Starting at $(Dates.now())"
@info """
Last commit:

$(cd(() -> read(`git log -n 1`, String), @__DIR__))
"""

########################################################################## #src

# ## Hardware selection

if CUDA.functional()
    ## For running on a CUDA compatible GPU
    @info "Running on CUDA"
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)
    clean() = (GC.gc(); CUDA.reclaim())
else
    ## For running on CPU.
    ## Consider reducing the sizes of DNS, LES, and CNN layers if
    ## you want to test run on a laptop.
    @warn "Running on CPU"
    backend = CPU()
    device = identity
    clean() = nothing
end

########################################################################## #src

# ## Data generation
#
# Create filtered DNS data for training, validation, and testing.

# Parameters
case = SymmetryClosure.testcase()

# DNS seeds
ntrajectory = 8
seeds = splitseed(case.seed_dns, ntrajectory)

# Create data
for (iseed, seed) in enumerate(seeds)
    if isnothing(taskid) || iseed == taskid
        @info "Creating DNS trajectory for seed $(repr(seed))"
    else
        # Each task does one initial condition
        @info "Skipping seed $(repr(seed)) for task $taskid"
        continue
    end
    filenames = map(Iterators.product(params.nles, params.filters)) do nles, Φ
        f = getdatafile(outdir, nles, Φ, seed)
        datadir = dirname(f)
        ispath(datadir) || mkpath(datadir)
        f
    end
    data = create_les_data(; case.params..., rng = Xoshiro(seed), filenames)
    @info(
        "Trajectory info:",
        data[1].comptime / 60,
        length(data[1].t),
        Base.summarysize(data) * 1e-9,
    )
end

# Computational time
docomp = false
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
