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

"""
    SteadyBodyForce(fu, fv, grid)

Two-dimensional steady body force `f(x, y) = [fu(x, y), fv(x, y)]`. 
"""
function SteadyBodyForce(fu, fv, grid)
    (; NV, indu, indv, xu, yu, xv, yv) = grid
    T = eltype(xu)
    F = zeros(T, NV)
    F[indu] .= reshape(fu.(xu, yu), :)
    F[indv] .= reshape(fv.(xv, yv), :)
    SteadyBodyForce(fu, fv, nothing, F)
end

"""
    SteadyBodyForce(fu, fv, fw, grid)

Three-dimensional steady body force `f(x, y, z) = [fu(x, y, z), fv(x, y, z), fw(x, y, z)]`. 
"""
function SteadyBodyForce(fu, fv, fw, grid)
    (; NV, indu, indv, indw, xu, yu, zu, xv, yv, zv, xw, yw, zw) = grid
    T = eltype(xu)
    F = zeros(T, NV)
    F[indu] .= reshape(fu.(xu, yu, zu), :)
    F[indv] .= reshape(fv.(xv, yv, zv), :)
    F[indw] .= reshape(fw.(xw, yw, zw), :)
    SteadyBodyForce(fu, fv, fw, F)
end

"""
    UnsteadyBodyForce{F1,F2,F3,T}

Unsteady (time-dependent) body force.
"""
Base.@kwdef mutable struct UnsteadyBodyForce{F1,F2,F3,T} <: AbstractBodyForce{T}
    fu::F1
    fv::F2
    fw::F3
    F::Vector{T} # For storing discrete body force
end

"""
    UnsteadyBodyForce(fu, fv, grid)

Two-dimensional unsteady body force `f(x, y, t) = [fu(x, y, t), fv(x, y, t)]`. 
"""
function UnsteadyBodyForce(fu, fv, grid)
    (; NV, xu) = grid
    T = eltype(xu)
    F = zeros(T, NV)
    UnteadyBodyForce(fu, fv, nothing, F)
end

"""
    UnsteadyBodyForce(fu, fv, fw, grid)

Three-dimensional unsteady body force `f(x, y, z, t) = [fu(x, y, z, t), fv(x, y, z, t),
fw(x, y, z, t)]`. 
"""
function UnsteadyBodyForce(fu, fv, fw, grid)
    (; NV, xu) = grid
    T = eltype(xu)
    F = zeros(T, NV)
    UnsteadyBodyForce(fu, fv, fw, F)
end
