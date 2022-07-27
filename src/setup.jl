"""
    Setup(; viscosity_model, grid, force, pressure_solver, bc)

Simulation setup.
"""
Base.@kwdef struct Setup{T,N,V<:AbstractViscosityModel{T},F<:AbstractBodyForce{T}}
    grid::Grid{T,N}
    operators::Operators{T}
    viscosity_model::V
    force::F
    function Setup(grid::Grid{T,N}, viscosity_model, force) where {T,N}
        operators = Operators{T}()
        new{T,N,typeof(viscosity_model),typeof(force)}(grid, operators, viscosity_model, force)
    end
end
