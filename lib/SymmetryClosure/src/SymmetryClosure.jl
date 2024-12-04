module SymmetryClosure

using StaticArrays
using ChainRulesCore
using KernelAbstractions
using LinearAlgebra
using IncompressibleNavierStokes
using NeuralClosure

include("tensor.jl")

export tensorbasis, tensorclosure, polynomial

end
