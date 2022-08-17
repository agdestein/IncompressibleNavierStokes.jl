"""
    Setup(; grid, operators, viscosity_model, force)

Simulation setup.
"""
Base.@kwdef struct Setup{T,N,V<:AbstractViscosityModel{T},F<:AbstractBodyForce{T}}
    grid::Grid{T,N}
    operators::Operators{T}
    viscosity_model::V
    force::F
end
