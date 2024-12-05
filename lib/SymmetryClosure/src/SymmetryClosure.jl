module SymmetryClosure

using StaticArrays
using ChainRulesCore
using KernelAbstractions
using LinearAlgebra
using IncompressibleNavierStokes
using NeuralClosure

include("tensorclosure.jl")

export tensorclosure, polynomial

end
