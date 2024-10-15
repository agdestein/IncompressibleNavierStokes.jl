"""
Utility functions for scripts.
"""
module PaperDC

using Dates
using DocStringExtensions
using EnumX
using LinearAlgebra
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    momentum!, divergence!, project!, apply_bc_u!, kinetic_energy!, scalewithvolume!
using JLD2
using LoggingExtras
using NeuralClosure
using Observables
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

include("observe.jl")
include("rk.jl")

export setsnelliuslogger
export namedtupleload
export observe_u, observe_v
export ProjectOrder, RKProject

end # module PaperDC
