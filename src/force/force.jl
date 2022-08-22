abstract type AbstractBodyForce{T} end

"""
    SteadyBodyForce{F1,F2,F3,T}

Steady (constant) body force.
"""
struct SteadyBodyForce{F1,F2,F3,T} <: AbstractBodyForce{T}
    fu::F1
    fv::F2
    fw::F3
    F::Vector{T} # For storing discrete body force
end

# 2D version
function SteadyBodyForce(fu, fv, grid::Grid{T,2}) where {T}
    (; NV, indu, indv, xu, yu, xv, yv) = grid
    F = zeros(T, NV) 
    F[indu] .= reshape(fu.(xu, yu), :)
    F[indv] .= reshape(fv.(xv, yv), :)
    SteadyBodyForce(fu, fv, nothing, F)
end

# 3D version
function SteadyBodyForce(fu, fv, fw, grid::Grid{T,3}) where {T}
    (; NV, indu, indv, indw, xu, yu, zu, xv, yv, zv, xw, yw, zw) = grid
    F = zeros(T, NV) 
    F[indu] .= reshape(fu.(xu, yu, zu), :)
    F[indv] .= reshape(fv.(xv, yv, zv), :)
    F[indw] .= reshape(fw.(xw, yw, zw), :)
    SteadyBodyForce(fu, fv, fw, F)
end

"""
    UnsteadyBodyForce{F1,F2,F3,T}

Forcing parameters with floating point type `T`.
"""
Base.@kwdef mutable struct UnsteadyBodyForce{F1,F2,F3,T} <: AbstractBodyForce{T}
    fu::F1
    fv::F2
    fw::F3
    F::Vector{T} # For storing discrete body force
end

# 2D version
function UnsteadyBodyForce(fu, fv, grid::Grid{T,2}) where {T}
    (; NV) = grid
    F = zeros(T, NV) 
    SteadyBodyForce(fu, fv, nothing, F)
end

# 3D version
function UnsteadyBodyForce(fu, fv, fw, grid::Grid{T,3}) where {T}
    (; NV) = grid
    F = zeros(T, NV) 
    SteadyBodyForce(fu, fv, fw, F)
end
