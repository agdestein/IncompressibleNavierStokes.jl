"""
    Setup(;
        grid,
        boundary_conditions,
        viscosity_model,
        convection_model = NoRegConvectionModel(),
        force,
        operators = Operators(grid, boundary_conditions, viscosity_model),
    )

Simulation setup.
"""
Base.@kwdef struct Setup{
    T,
    N,
    V<:AbstractViscosityModel{T},
    C<:AbstractConvectionModel,
    F<:AbstractBodyForce{T},
}
    grid::Grid{T,N}
    boundary_conditions::BoundaryConditions{T}
    viscosity_model::V
    convection_model::C = NoRegConvectionModel()
    force::F
    operators::Operators{T} = Operators(grid, boundary_conditions, viscosity_model)
end
