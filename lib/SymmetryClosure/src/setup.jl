# Some script utils

function slurm_vars()
    jobid = haskey(ENV, "SLURM_JOB_ID") ? parse(Int, ENV["SLURM_JOB_ID"]) : nothing
    taskid =
        haskey(ENV, "SLURM_ARRAY_TASK_ID") ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) :
        nothing
    isnothing(jobid) || @info "Running on SLURM" jobid taskid
    (; jobid, taskid)
end

function time_info()
    @info "Starting at $(Dates.now())"
    @info """
    Last commit:

    $(cd(() -> read(`git log -n 1`, String), @__DIR__))
    """
end

hardware() =
    if CUDA.functional()
        @info "Running on CUDA"
        CUDA.allowscalar(false)
        backend = CUDABackend()
        device = x -> adapt(backend, x)
        clean = () -> (GC.gc(); CUDA.reclaim())
        (; backend, device, clean)
    else
        @warn """
        Running on CPU.
        Consider reducing the size of DNS, LES, and CNN layers if
        you want to test run on a laptop.
        """
        (; backend = CPU(), device = identity, clean = () -> nothing)
    end

function splatfileparts(args...; kwargs...)
    sargs = string.(args)
    skwargs = map((k, v) -> string(k) * "=" * string(v), keys(kwargs), values(kwargs))
    s = [sargs..., skwargs...]
    join(s, "_")
end

getdatafile(outdir, nles, filter, seed) =
    joinpath(outdir, "data", splatfileparts(; seed = repr(seed), filter, nles) * ".jld2")

function namedtupleload(file)
    dict = load(file)
    k, v = keys(dict), values(dict)
    pairs = @. Symbol(k) => v
    (; pairs...)
end

getsetup(; params, nles) = Setup(;
    x = ntuple(α -> range(params.lims..., nles + 1), params.D),
    params.Re,
    params.backend,
    params.bodyforce,
    params.issteadybodyforce,
)
