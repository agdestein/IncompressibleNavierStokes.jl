"""
    Setup(; viscosity_model, convection_model, grid, force, bc)

Simulation setup.
"""
Base.@kwdef struct Setup{T,N}
    viscosity_model::AbstractViscosityModel{T}
    convection_model::AbstractConvectionModel{T}
    grid::Grid{T,N}
    operators::Operators{T} = Operators{T}()
    force::AbstractBodyForce{T}
    bc::BC{T}
end
