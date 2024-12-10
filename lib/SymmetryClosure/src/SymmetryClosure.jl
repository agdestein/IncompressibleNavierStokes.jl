module SymmetryClosure

using Accessors
using Adapt
using ChainRulesCore
using CUDA
using Dates
using IncompressibleNavierStokes
using JLD2
using KernelAbstractions
using LinearAlgebra
using Lux
using NeuralClosure
using NNlib
using Optimisers
using Random
using StaticArrays

include("tensorclosure.jl")
include("setup.jl")
include("cases.jl")
include("train.jl")

export tensorclosure, polynomial, create_cnn
export slurm_vars,
    time_info,
    hardware,
    splatfileparts,
    getdatafile,
    namedtupleload,
    splitseed,
    getsetup,
    testcase
export trainpost

end
