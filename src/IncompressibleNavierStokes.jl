"Incompressible Navier-Stokes solver"
module IncompressibleNavierStokes

using SparseArrays
using UnPack

# Spatial
include("spatial/momentum.jl")
include("spatial/check_conservation.jl")
include("spatial/boundary_conditions/set_bc_vectors.jl")
include("spatial/convection.jl")
include("spatial/diffusion.jl")
include("spatial/strain_tensor.jl")
include("spatial/turbulent_viscosity.jl")
include("spatial/operators/interpolate_nu.jl")

# Bodyforce
include("bodyforce/force.jl")

# Unsteady
include("solvers/set_timestep.jl")

# Solvers
include("solvers/solve_steady.jl")

# Utils
include("utils/filter_convection.jl")

# Solvers
export solve_steady

end
