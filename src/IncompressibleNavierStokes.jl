"Incompressible Navier-Stokes solver"
module IncompressibleNavierStokes

using SparseArrays
using UnPack

# Setup
include("parameters.jl")

# Preprocess
include("preprocess/check_input.jl")

# Spatial
include("spatial/check_conservation.jl")
include("spatial/convection.jl")
include("spatial/diffusion.jl")
include("spatial/momentum.jl")
include("spatial/strain_tensor.jl")
include("spatial/turbulent_viscosity.jl")
include("spatial/boundary_conditions/BC_general.jl")
include("spatial/boundary_conditions/BC_general_stag.jl")
include("spatial/boundary_conditions/set_bc_vectors.jl")
include("spatial/operators/interpolate_nu.jl")
include("spatial/operators/operator_averaging.jl")
include("spatial/operators/operator_convection_diffusion.jl")
include("spatial/operators/operator_divergence.jl")
include("spatial/operators/operator_interpolation.jl")
include("spatial/operators/operator_mesh.jl")
include("spatial/operators/operator_postprocessing.jl")
include("spatial/operators/operator_regularization.jl")
include("spatial/operators/operator_turbulent_diffusion.jl")

# Bodyforce
include("bodyforce/force.jl")

# Unsteady
include("solvers/set_timestep.jl")

# Solvers
include("solvers/solve_steady.jl")
include("solvers/set_timestep.jl")
include("solvers/pressure/make_cholinc.jl")

# Utils
include("utils/filter_convection.jl")

# Setup
export Case, Fluid, Visc, Grid, Discretization, Force, ROM, IBM, Time, SolverSettings, Visualization

# Solvers
export solve_steady

end
