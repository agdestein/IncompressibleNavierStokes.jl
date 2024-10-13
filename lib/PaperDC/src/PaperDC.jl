"""
Utility functions for scripts.
"""
module PaperDC

using DocStringExtensions
using EnumX
using LinearAlgebra
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    momentum!, divergence!, project!, apply_bc_u!, kinetic_energy!, scalewithvolume!
using JLD2
using NeuralClosure
using Observables
using Random

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

export namedtupleload
export observe_u, observe_v
export ProjectOrder, RKProject

end # module PaperDC
