"Incompressible Navier-Stokes solvers"
module IncompressibleNavierStokes

using LinearAlgebra: Diagonal, Factorization, I, cholesky, factorize, ldiv!, mul!
using SparseArrays: SparseMatrixCSC, blockdiag, sparse, spdiagm, spzeros
using UnPack: @pack!, @unpack
# Using Plots: contour, contourf, title!
using Makie:
    Axis, Colorbar, DataAspect, Figure, Node, contourf, contourf!, limits!, lines!, record

# Setup
include("solvers/time/runge_kutta_methods.jl")
include("parameters.jl")

# Preprocess
include("preprocess/check_input.jl")
include("preprocess/create_mesh.jl")
include("preprocess/create_initial_conditions.jl")

# Momentum equation
include("momentum/momentumcache.jl")
include("momentum/bodyforce.jl")
include("momentum/check_conservation.jl")
include("momentum/check_symmetry.jl")
include("momentum/convection.jl")
include("momentum/diffusion.jl")
include("momentum/momentum.jl")
include("momentum/strain_tensor.jl")
include("momentum/turbulent_viscosity.jl")

# Boundary condtions
include("boundary_conditions/bc_diff_stag.jl")
include("boundary_conditions/bc_diff_stag3.jl")
include("boundary_conditions/bc_general.jl")
include("boundary_conditions/bc_general_stag.jl")
include("boundary_conditions/create_boundary_conditions.jl")
include("boundary_conditions/set_bc_vectors.jl")

# Grid
include("grid/nonuniform_grid.jl")

# Operators
include("operators/build_operators.jl")
include("operators/interpolate_nu.jl")
include("operators/operator_averaging.jl")
include("operators/operator_convection_diffusion.jl")
include("operators/operator_divergence.jl")
include("operators/operator_interpolation.jl")
include("operators/operator_mesh.jl")
include("operators/operator_postprocessing.jl")
include("operators/operator_regularization.jl")
include("operators/operator_turbulent_diffusion.jl")

# Reduced Order Model
include("rom/momentum_rom.jl")

# Solvers
include("solvers/get_timestep.jl")
include("solvers/solve_steady.jl")
include("solvers/solve_steady_ke.jl")
include("solvers/solve_steady_ibm.jl")
include("solvers/solve_unsteady.jl")
include("solvers/solve_unsteady_ke.jl")
include("solvers/solve_unsteady_rom.jl")

include("solvers/pressure/pressure_poisson.jl")
include("solvers/pressure/pressure_additional_solve.jl")

include("solvers/time/step_AB_CN.jl")
include("solvers/time/step_ERK.jl")
include("solvers/time/step_ERK_ROM.jl")
include("solvers/time/step_IRK.jl")
include("solvers/time/step_IRK_ROM.jl")

# Utils
include("utils/filter_convection.jl")

# Postprocess
include("postprocess/postprocess.jl")
include("postprocess/get_velocity.jl")
include("postprocess/get_vorticity.jl")
include("postprocess/get_streamfunction.jl")
include("postprocess/plot_vorticity.jl")
include("postprocess/plot_pressure.jl")
include("postprocess/plot_streamfunction.jl")

# Main driver
include("main.jl")

# Reexport
export @unpack

# Setup
export Case,
    Fluid,
    Visc,
    Grid,
    Discretization,
    Force,
    ROM,
    IBM,
    Time,
    SolverSettings,
    Visualization,
    BC,
    Setup

# Spatial
export nonuniform_grid

# Main driver
export main
export create_mesh!,
    create_boundary_conditions!,
    build_operators!,
    create_initial_conditions,
    set_bc_vectors!,
    force,
    check_input!,
    solve_steady_ke!,
    solve_steady!,
    solve_steady_ibm!,
    solve_unsteady_ke!,
    solve_unsteady_rom!,
    solve_unsteady!

export postprocess, plot_pressure, plot_streamfunction, plot_vorticity

# Runge Kutta methods

# Explicit Methods
export FE11, SSP22, SSP42, SSP33, SSP43, SSP104, rSSPs2, rSSPs3, Wray3, RK56, DOPRI6

# Implicit Methods
export BE11, SDIRK34, ISSPm2, ISSPs3

# Half explicit methods
export HEM3, HEM3BS, HEM5

# Classical Methods
export GL1, GL2, GL3, RIA1, RIA2, RIA3, RIIA1, RIIA2, RIIA3, LIIIA2, LIIIA3

# Chebyshev methods
export CHDIRK3, CHCONS3, CHC3, CHC5

# Miscellaneous Methods
export Mid22, MTE22, CN22, Heun33, RK33C2, RK33P2, RK44, RK44C2, RK44C23, RK44P2

# DSRK Methods
export DSso2, DSRK2, DSRK3

# "Non-SSP" Methods of Wong & Spiteri
export NSSP21, NSSP32, NSSP33, NSSP53

end
