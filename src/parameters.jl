"""
    SolverSettings(;
        pressure_solver = DirectPressureSolver(),
        p_add_solve = true,
        abstol = 1e-10,
        reltol = 1e-14,
        maxiter = 10,
        newton_type = :approximate,
    )

Solver settings.
"""
Base.@kwdef mutable struct SolverSettings{T}
    pressure_solver::PressureSolver = DirectPressureSolver() # PressureSolver
    p_add_solve::Bool = true                                 # Additional pressure solve to make it same order as velocity

    abstol::T = 1e-14                                        # Absolute accuracy
    reltol::T = 1e-14                                        # Relative accuracy
    maxiter::Int = 10                                        # Maximum number of iterations
    # :no: Replace iteration matrix with I/Δt (no Jacobian)
    # :approximate: Build Jacobian once before iterations only
    # :full: Build Jacobian at each iteration
    newton_type::Symbol = :approximate
    newton_factor::Bool = false                              # Newton factor
end

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
