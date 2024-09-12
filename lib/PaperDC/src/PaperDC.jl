"""
Utility functions for scripts.
"""
module PaperDC

using DocStringExtensions
using EnumX
using IncompressibleNavierStokes
using IncompressibleNavierStokes: momentum!, divergence!, project!, apply_bc_u!
using Observables
using LinearAlgebra

# Inherit docstring templates
@template (MODULES, FUNCTIONS, METHODS, TYPES) = IncompressibleNavierStokes

include("observe.jl")
include("rk.jl")

export observe_u, observe_v
export ProjectOrder, RKProject

end # module PaperDC
