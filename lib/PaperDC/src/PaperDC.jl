"""
Utility functions for scripts.
"""
module PaperDC

using Accessors
using Adapt
using Dates
using DocStringExtensions
using EnumX
using LinearAlgebra
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    momentum!, divergence!, project!, apply_bc_u!, kinetic_energy!, scalewithvolume!
using JLD2
using LoggingExtras
using Lux
using NeuralClosure
using Observables
using Optimisers
using Random

"Write output to file, as the default SLURM file is not updated often enough."
function setsnelliuslogger(logfile)
    filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
    logger = TeeLogger(ConsoleLogger(), filelogger)
    oldlogger = global_logger(logger)
    @info """
    Logging to file: $logfile

    Starting at $(Dates.now()).

    Last commit:

    $(cd(() -> read(`git log -n 1`, String), @__DIR__))
    """
    oldlogger
end

# Inherit docstring templates
@template (MODULES, FUNCTIONS, METHODS, TYPES) = IncompressibleNavierStokes

"Load JLD2 file as named tuple (instead of dict)."
function namedtupleload(file)
    dict = load(file)
    k, v = keys(dict), values(dict)
    pairs = @. Symbol(k) => v
    (; pairs...)
end

"""
Make file name from parts.

```@example
julia> splatfileparts("toto", 23; haha = 1e3, hehe = "hihi")
"toto_23_haha=1000.0_hehe=hihi"
```
"""
function splatfileparts(args...; kwargs...)
    sargs = string.(args)
    skwargs = map((k, v) -> string(k) * "=" * string(v), keys(kwargs), values(kwargs))
    s = [sargs..., skwargs...]
    join(s, "_")
end

getsetup(; params, nles) = Setup(;
    x = ntuple(α -> range(params.lims..., nles + 1), params.D),
    params.Re,
    params.ArrayType,
    params.bodyforce,
    params.issteadybodyforce,
)

include("observe.jl")
include("rk.jl")
include("train.jl")

export setsnelliuslogger
export namedtupleload, splatfileparts
export observe_u, observe_v
export ProjectOrder, RKProject
export getdatafile, createdata, getsetup
export trainprior, loadprior
export trainpost, loadpost
export trainsmagorinsky, loadsmagorinsky

end # module PaperDC
