"""
    PressureSolver

Pressure solver for the Poisson equation.
"""
abstract type AbstractPressureSolver{T} end

"""
    DirectPressureSolver()

Direct pressure solver using a LU decomposition.
"""
Base.@kwdef mutable struct DirectPressureSolver{T} <: AbstractPressureSolver{T}
    A_fact::Factorization{T} = cholesky(spzeros(T, 0, 0))
end

"""
    CGPressureSolver(abstol, reltol, maxiter)

Conjugate gradients iterative pressure solver.
"""
Base.@kwdef mutable struct CGPressureSolver{T} <: AbstractPressureSolver{T}
    A::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    abstol::T = 0
    reltol::T = √eps(T)
    maxiter::Int = 0
end

"""
    FourierPressureSolver()

Fourier transform pressure solver for periodic domains.
"""
Base.@kwdef mutable struct FourierPressureSolver{T} <: AbstractPressureSolver{T}
    # TODO: Pass dimensionality to create concrete types
    Â::Array{Complex{T}} = Complex{T}[]
    p̂::Array{Complex{T}} = Complex{T}[]
    f̂::Array{Complex{T}} = Complex{T}[]
end
