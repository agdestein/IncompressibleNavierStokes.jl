"Incompressible Navier-Stokes solver"
module IncompressibleNavierStokes

using LinearAlgebra: I
using SparseArrays: SparseMatrixCSC, sparse, spdiagm, spzeros
using UnPack: @unpack
using Plots: contour, contourf

# Setup
include("parameters.jl")

# Preprocess
include("preprocess/check_input.jl")
include("preprocess/create_mesh.jl")

# Spatial
include("spatial/check_conservation.jl")
include("spatial/check_symmetry.jl")
include("spatial/convection.jl")
include("spatial/create_initial_conditions.jl")
include("spatial/diffusion.jl")
include("spatial/momentum.jl")
include("spatial/strain_tensor.jl")
include("spatial/turbulent_viscosity.jl")

include("spatial/boundary_conditions/bc_diff_stag.jl")
include("spatial/boundary_conditions/bc_diff_stag3.jl")
include("spatial/boundary_conditions/bc_general.jl")
include("spatial/boundary_conditions/bc_general_stag.jl")
include("spatial/boundary_conditions/create_boundary_conditions.jl")
include("spatial/boundary_conditions/set_bc_vectors.jl")

include("spatial/grid/nonuniform_grid.jl")

include("spatial/operators/build_operators.jl")
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

# Solvers
include("solvers/get_timestep.jl")
include("solvers/solve_steady.jl")
include("solvers/solve_steady_ke.jl")
include("solvers/solve_steady_ibm.jl")
include("solvers/solve_unsteady.jl")
include("solvers/solve_unsteady_ke.jl")
include("solvers/solve_unsteady_rom.jl")

# Utils
include("utils/filter_convection.jl")

# Postprocess
include("postprocess/postprocess.jl")
include("postprocess/get_vorticity.jl")
include("postprocess/get_streamfunction.jl")

# Main driver
include("main.jl")

# Reexport
export @unpack

# Setup
export Case, Fluid, Visc, Grid, Discretization, Force, ROM, IBM, Time, SolverSettings, Visualization, BC, Setup

# Spatial
export nonuniform_grid

# Main driver
export main


end
