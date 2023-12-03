"""
    IncompressibleNavierStokes

Energy-conserving solvers for the incompressible Navier-Stokes equations.
"""
module IncompressibleNavierStokes

using Adapt
using ComponentArrays: ComponentArray
using FFTW
using IterativeSolvers
using KernelAbstractions
using LinearAlgebra
using Lux
using Makie
using NNlib
using Optimisers
using Printf
using Random
using SparseArrays
using StaticArrays
using Statistics
using Tullio
using WriteVTK: CollectionFile, paraview_collection, vtk_grid, vtk_save
using Zygote

# Must be loaded inside for Tullio to work correctly
using CUDA

# # Easily retrieve value from Val
# (::Val{x})() where {x} = x

# Boundary conditions
include("boundary_conditions.jl")

# Grid
include("grid/dimension.jl")
include("grid/grid.jl")
include("grid/stretched_grid.jl")
include("grid/cosine_grid.jl")
include("grid/max_size.jl")

# Models
include("models/viscosity_models.jl")

# Setup
include("setup.jl")

# Pressure solvers
include("solvers/pressure/solvers.jl")
include("solvers/pressure/poisson.jl")
include("solvers/pressure/pressure.jl")

# Time steppers
include("time_steppers/methods.jl")
include("time_steppers/tableaux.jl")
include("time_steppers/nstage.jl")
include("time_steppers/time_stepper_caches.jl")
include("time_steppers/step.jl")
include("time_steppers/isexplicit.jl")
include("time_steppers/lambda_max.jl")

# Preprocess
include("create_initial_conditions.jl")

# Processors
include("processors/processors.jl")
include("processors/real_time_plot.jl")
include("processors/animator.jl")

# Discrete operators
include("operators.jl")
include("filter.jl")

# Solvers
include("solvers/get_timestep.jl")
include("solvers/solve_steady_state.jl")
include("solvers/solve_unsteady.jl")

# Utils
include("utils/plotgrid.jl")
include("utils/save_vtk.jl")
include("utils/get_lims.jl")
include("utils/plotmat.jl")

# Closure models
include("closures/closure.jl")
include("closures/cnn.jl")
include("closures/fno.jl")
include("closures/training.jl")
include("closures/create_les_data.jl")

# Boundary conditions
export PeriodicBC, DirichletBC, SymmetricBC, PressureBC

# Models
export LaminarModel, MixingLengthModel, SmagorinskyModel, QRModel

# Processors
export processor, timelogger, vtk_writer, fieldsaver, realtimeplotter
export fieldplot, energy_history_plot, energy_spectrum_plot
export animator

# Setup
export Setup

# 1D grids
export stretched_grid, cosine_grid

# Pressure solvers
export DirectPressureSolver, CGPressureSolver, SpectralPressureSolver

# Solvers
export solve_unsteady, solve_steady_state

export create_initial_conditions, random_field

export plotgrid, save_vtk
export plotmat

# Closure models
export cnn, fno, FourierLayer
export train
export mean_squared_error, relative_error
export createloss, createdataloader, create_callback, create_les_data, create_io_arrays
export wrappedclosure

# ODE methods

export AdamsBashforthCrankNicolsonMethod, OneLegMethod

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
