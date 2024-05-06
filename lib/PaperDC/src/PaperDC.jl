"""
Utility functions for scripts.
"""
module PaperDC

using IncompressibleNavierStokes
using IncompressibleNavierStokes: momentum!, divergence!, project!, apply_bc_u!
using Observables
using LinearAlgebra

include("observe.jl")
include("rk.jl")

export observe_u, observe_v
export RKProject

end # module PaperDC
