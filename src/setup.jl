"""
    Setup(; grid, operators, bc, viscosity_model, convection_model, force)

Simulation setup.
"""
Base.@kwdef struct Setup{
    T,
    N,
    V<:AbstractViscosityModel{T},
    C<:AbstractConvectionModel{T},
    F<:AbstractBodyForce{T},
}
    grid::Grid{T,N}
    bc::BC{T}
    viscosity_model::V
    convection_model::C
    force::F
    operators::Operators{T} = Operators(grid, bc, viscosity_model)
end
