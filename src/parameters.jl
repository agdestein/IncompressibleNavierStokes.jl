# Case information
Base.@kwdef mutable struct Case
    name::String = "example"                 # Case name
    problem::Problem = UnsteadyProblem()     # Problem type
    regularization::String = "no"            # Convective term regularization: "no", "leray", "C2", "C4"
    initial_velocity_u::Function = () -> error("initial_velocity_u not implemented")
    initial_velocity_v::Function = () -> error("initial_velocity_v not implemented")
    initial_velocity_w::Function = () -> error("initial_velocity_w not implemented")
    initial_pressure::Function = () -> error("initial_pressure not implemented")
end

# Time stepping
Base.@kwdef mutable struct Time{T}
    t_start::T = 0                                           # Start time
    t_end::T = 1                                             # End time
    Δt::T = (t_end - t_start) / 100                          # Timestep
    method::AbstractODEMethod = RK44()                       # ODE method
    method_startup::AbstractODEMethod = RK44()               # Startup method for methods that are not self starting
    nstartup::Int = 0                                        # Number of velocity fields necessary for start-up = equal to order of method
    isadaptive::Bool = false                                 # Adapt timestep every n_adapt_Δt iterations
    n_adapt_Δt::Int = 1                                      # Number of iterations between timestep adjustment
    CFL::T = 1 // 2                                          # CFL number for adaptive methods
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
Base.@kwdef struct Setup{T, N}
    case::Case = Case()
    model::AbstractViscosityModel{T} = LaminarModel{T}()
    grid::Grid{T, N} = Grid{T, N}()
    operators::Operators{T} = Operators{T}()
    force::AbstractBodyForce{T} = SteadyBodyForce{T}()
    time::Time{T} = Time{T}()
    solver_settings::SolverSettings{T} = SolverSettings{T}()
    processors::Vector{Processor} = Processor[]
    bc::BC{T} = BC{T}()
end
