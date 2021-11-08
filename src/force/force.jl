abstract type AbstractBodyForce{T} end

"""
    SteadyBodyForce{T}

Steady (constant) body force.
"""
Base.@kwdef mutable struct SteadyBodyForce{T} <: AbstractBodyForce{T}
    bodyforce_u::Function = (x, y) -> 0
    bodyforce_v::Function = (x, y) -> 0
    F::Vector{T} = T[] # For storing constant body force
end

"""
    UnsteadyBodyForce{T}

Forcing parameters with floating point type `T`.
"""
Base.@kwdef mutable struct UnsteadyBodyForce{T} <: AbstractBodyForce{T}
    F::Vector{T} = T[] # For storing constant body force
    bodyforce_u::Function = () -> error("bodyforce_x not implemented")
    bodyforce_v::Function = () -> error("bodyforce_y not implemented")
end



