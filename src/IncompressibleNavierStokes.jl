"Incompressible Navier-Stokes solvers"
module IncompressibleNavierStokes

using FFTW: fft, ifft
using IterativeSolvers: cg!
using LinearAlgebra: Diagonal, Factorization, I, cholesky, factorize, ldiv!, lu, mul!
using SparseArrays: SparseMatrixCSC, blockdiag, nnz, sparse, spdiagm, spzeros
using UnPack: @pack!, @unpack
# Using Plots: contour, contourf, title!
using Makie:
    Axis,
    Colorbar,
    DataAspect,
    Figure,
    Node,
    contour!,
    contourf,
    contourf!,
    limits!,
    lines!,
    record,
    save

# Models
include("models/models.jl")

# Problems
include("problems/problems.jl")
include("problems/is_steady.jl")

# Types
include("time_steppers/time_steppers.jl")
include("time_steppers/tableaus.jl")
include("time_steppers/nstage.jl")
include("time_steppers/time_stepper_caches.jl")
include("time_steppers/step.jl")
include("time_steppers/step_ab_cn.jl")
include("time_steppers/step_erk.jl")
include("time_steppers/step_erk_rom.jl")
include("time_steppers/step_irk.jl")
include("time_steppers/step_irk_rom.jl")
include("time_steppers/step_one_leg.jl")
include("time_steppers/isexplicit.jl")
include("time_steppers/needs_startup_stepper.jl")
include("time_steppers/lambda_max.jl")
include("solvers/pressure/pressure_solvers.jl")

include("parameters.jl")

# Preprocess
include("preprocess/check_input.jl")
include("preprocess/create_mesh.jl")
include("preprocess/create_initial_conditions.jl")

# Momentum equation
include("momentum/momentumcache.jl")
include("momentum/bodyforce.jl")
include("momentum/compute_conservation.jl")
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
include("solvers/real_time_plot/initialize_rtp.jl")
include("solvers/real_time_plot/update_rtp.jl")
include("solvers/get_timestep.jl")
include("solvers/solve.jl")

include("solvers/pressure/pressure_poisson.jl")
include("solvers/pressure/pressure_additional_solve.jl")

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

# Problems
export SteadyStateProblem, UnsteadyProblem, is_steady

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

# Pressure solvers
export DirectPressureSolver, CGPressureSolver, FourierPressureSolver

# Main driver
export main
export create_mesh!,
    create_boundary_conditions!,
    build_operators!,
    create_initial_conditions,
    set_bc_vectors!,
    force,
    check_input!,
    solve

export postprocess, plot_pressure, plot_streamfunction, plot_vorticity

export AdamsBashforthCrankNicolsonStepper, OneLegStepper

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
