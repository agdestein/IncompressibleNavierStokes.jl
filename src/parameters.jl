# Case information
Base.@kwdef mutable struct Case
    initial_velocity_u::Function = () -> error("initial_velocity_u not implemented")
    initial_velocity_v::Function = () -> error("initial_velocity_v not implemented")
    initial_velocity_w::Function = () -> error("initial_velocity_w not implemented")
    initial_pressure::Function = () -> error("initial_pressure not implemented")
end

# Solver settings
Base.@kwdef mutable struct SolverSettings{T}
    pressure_solver::PressureSolver = DirectPressureSolver() # PressureSolver
    p_initial::Bool = true                                   # Calculate compatible IC for the pressure
    p_add_solve::Bool = true                                 # Additional pressure solve to make it same order as velocity

    # Accuracy for non-linear solves (method 62 = 72 = 9)
    nonlinear_acc::T = 1e-14                                 # Absolute accuracy
    nonlinear_relacc::T = 1e-14                              # Relative accuracy
    nonlinear_maxit::Int = 10                                # Maximum number of iterations

    # "no": Replace iteration matrix with I/Δt (no Jacobian)
    # "approximate": Build Jacobian once before iterations only
    # "full": Build Jacobian at each iteration
    nonlinear_Newton::String = "approximate"

    Jacobian_type::String = "picard"                         # "picard": Picard linearization, "newton": Newton linearization
    Newton_factor::Bool = false                              # Newton factor
    nonlinear_startingvalues::Bool = false                   # Extrapolate values from last time step to get accurate initial guess (for `UnsteadyProblem`s only)
    nPicard::Int = 5                                         # Number of Picard steps before switching to Newton when linearization is Newton (for `SteadyStateProblem`s only)
end

# Setup
Base.@kwdef struct Setup{T,N}
    case::Case
    viscosity_model::AbstractViscosityModel{T}
    convection_model::AbstractConvectionModel{T}
    grid::Grid{T,N}
    operators::Operators{T} = Operators{T}()
    force::AbstractBodyForce{T}
    solver_settings::SolverSettings{T}
    bc::BC{T}
end
