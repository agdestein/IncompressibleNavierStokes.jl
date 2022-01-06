"""
    IncompressibleNavierStokes

Energy-conserving solvers for the incompressible Navier-Stokes equations.
"""
module IncompressibleNavierStokes

using FFTW: fft!, ifft!
using Interpolations: LinearInterpolation
using IterativeSolvers: cg!
using LinearAlgebra: Diagonal, Factorization, UpperTriangular, I, cholesky, factorize, ldiv!, lu, mul!
using SparseArrays: SparseMatrixCSC, blockdiag, nnz, sparse, spdiagm, spzeros
using UnPack: @pack!, @unpack
using WriteVTK: CollectionFile, paraview_collection, vtk_grid, vtk_save
using Makie


# Convenience notation
const ⊗ = kron


# Grid
include("grid/grid.jl")
include("grid/nonuniform_grid.jl")
include("grid/create_grid.jl")

# Force
include("force/force.jl")
include("force/build_force.jl")

# Models
include("models/models.jl")

# Problems
include("problems/problems.jl")
include("problems/is_steady.jl")

# Boundary condtions
include("boundary_conditions/boundary_conditions.jl")
include("boundary_conditions/create_boundary_conditions.jl")
include("boundary_conditions/bc_diff_stag.jl")
include("boundary_conditions/bc_diff_stag3.jl")
include("boundary_conditions/bc_general.jl")
include("boundary_conditions/bc_general_stag.jl")
include("boundary_conditions/set_bc_vectors.jl")

# Operators
include("operators/operators.jl")
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
include("operators/ke_production.jl")
include("operators/ke_convection.jl")
include("operators/ke_diffusion.jl")
include("operators/ke_viscosity.jl")
include("operators/operator_viscosity.jl")

# Preprocess
include("preprocess/create_initial_conditions.jl")

# Processors
include("processors/processors.jl")
include("processors/initialize.jl")
include("processors/process.jl")
include("processors/finalize.jl")

# Types
include("time_steppers/methods.jl")
include("time_steppers/tableaux.jl")
include("time_steppers/nstage.jl")
include("time_steppers/time_stepper_caches.jl")
include("solvers/pressure/pressure_solvers.jl")
include("momentum/momentumcache.jl")
include("parameters.jl")

# Time steppers
include("time_steppers/time_steppers.jl")
include("time_steppers/change_time_stepper.jl")
include("time_steppers/step.jl")
include("time_steppers/isexplicit.jl")
include("time_steppers/needs_startup_method.jl")
include("time_steppers/lambda_max.jl")

# Momentum equation
include("momentum/bodyforce.jl")
include("momentum/compute_conservation.jl")
include("momentum/check_symmetry.jl")
include("momentum/convection.jl")
include("momentum/diffusion.jl")
include("momentum/momentum.jl")
include("momentum/strain_tensor.jl")
include("momentum/turbulent_K.jl")
include("momentum/turbulent_viscosity.jl")

# Solvers
include("solvers/get_timestep.jl")
include("solvers/solve.jl")

include("solvers/pressure/pressure_poisson.jl")
include("solvers/pressure/pressure_additional_solve.jl")

# Utils
include("utils/filter_convection.jl")

# Postprocess
include("postprocess/get_velocity.jl")
include("postprocess/get_vorticity.jl")
include("postprocess/get_streamfunction.jl")
include("postprocess/plot_vorticity.jl")
include("postprocess/plot_pressure.jl")
include("postprocess/plot_streamfunction.jl")
include("postprocess/plot_tracers.jl")

# Reexport
export @pack!, @unpack

# Grid
export create_grid

# Force
export SteadyBodyForce, UnsteadyBodyForce

# Models
export LaminarModel, KEpsilonModel, MixingLengthModel, SmagorinskyModel, QRModel

# Processors
export Logger, RealTimePlotter, VTKWriter, QuantityTracer

# Problems
export SteadyStateProblem, UnsteadyProblem, is_steady

# Setup
export Case, Grid, Operators, Time, SolverSettings, BC, Setup

# Spatial
export nonuniform_grid

# Pressure solvers
export DirectPressureSolver, CGPressureSolver, FourierPressureSolver

# Main driver
export create_boundary_conditions,
    build_operators!,
    create_initial_conditions,
    set_bc_vectors!,
    solve,
    get_velocity

export momentum!

export plot_pressure, plot_streamfunction, plot_vorticity, plot_tracers

# ODE methods

export AdamsBashforthCrankNicolsonMethod, OneLegMethod

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
