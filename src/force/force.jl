abstract type AbstractBodyForce{T} end

"""
    SteadyBodyForce{T}

Steady (constant) body force.
"""
Base.@kwdef mutable struct SteadyBodyForce{T} <: AbstractBodyForce{T}
    bodyforce_u::Function = (x, y, z) -> 0
    bodyforce_v::Function = (x, y, z) -> 0
    bodyforce_w::Function = (x, y, z) -> 0
    F::Vector{T} = T[] # For storing constant body force
end

"""
    UnsteadyBodyForce{T}

Forcing parameters with floating point type `T`.
"""
Base.@kwdef mutable struct UnsteadyBodyForce{T} <: AbstractBodyForce{T}
    F::Vector{T} = T[] # For storing constant body force
    bodyforce_u::Function = (x, y, z, t) -> error("bodyforce_u not implemented")
    bodyforce_v::Function = (x, y, z, t) -> error("bodyforce_v not implemented")
    bodyforce_w::Function = (x, y, z, t) -> error("bodyforce_w not implemented")
end
