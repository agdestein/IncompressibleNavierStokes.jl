"""
    Setup(; viscosity_model, convection_model, grid, force, solver_settings, bc)

Simulation setup.
"""
Base.@kwdef struct Setup{T,N}
    viscosity_model::AbstractViscosityModel{T}
    convection_model::AbstractConvectionModel{T}
    grid::Grid{T,N}
    operators::Operators{T} = Operators{T}()
    force::AbstractBodyForce{T}
    solver_settings::SolverSettings{T}
    bc::BC{T}
end
