"""
    Setup(; viscosity_model, grid, force, pressure_solver, bc)

Simulation setup.
"""
Base.@kwdef struct Setup{T,N}
    viscosity_model::AbstractViscosityModel{T}
    grid::Grid{T,N}
    operators::Operators{T} = Operators{T}()
    force::AbstractBodyForce{T}
    pressure_solver::AbstractPressureSolver{T}
end
