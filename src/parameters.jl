# Case information
Base.@kwdef mutable struct Case
    name::String = "example"                 # Case name
    problem::Problem = UnsteadyProblem()     # Problem type
    regularization::String = "no"            # Convective term regularization: "no", "leray", "C2", "C4"
    ibm::Bool = false                        # Use immersed boundary method
    initial_velocity_u::Function = () -> error("initial_velocity_u not implemented")
    initial_velocity_v::Function = () -> error("initial_velocity_v not implemented")
    initial_velocity_w::Function = () -> error("initial_velocity_w not implemented")
    initial_pressure::Function = () -> error("initial_pressure not implemented")
end

# Physical properties
Base.@kwdef mutable struct Fluid{T}
    Re::T = 1                                # Reynolds number
    U1::T = 1                                # Velocity scales
    U2::T = 1                                # Velocity scales
    d_layer::T = 1                           # Thickness of layer
end

# Rom parameters
Base.@kwdef mutable struct ROM
    use_rom::Bool = false                                    # Use reduced order model
    rom_type::String = "POD"                                 # "POD",  "Fourier"
    M::Int = 10                                              # Number of velocity modes for reduced order model
    Mp::Int = 10                                             # Number of pressure modes for reduced order model
    precompute_convection::Bool = true                       # Precomputed convection matrices
    precompute_diffusion::Bool = true                        # Precomputed diffusion matrices
    precompute_force::Bool = true                            # Precomputed forcing term
    t_snapshots::Int = 0                                     # Snapshots
    Δt_snapshots::Bool = false
    mom_cons::Bool = false                                   # Momentum conserving SVD
    rom_bc::Int = 0                                          # 0: homogeneous (no-slip = periodic) 1: non-homogeneous = time-independent 2: non-homogeneous = time-dependent
    weighted_norm::Bool = true                               # Use weighted norm (using finite volumes as weights)
    pressure_recovery::Bool = false                          # False: no pressure computation, true: compute pressure with PPE-ROM
    pressure_precompute::Int = 0                             # In case of pressure_recovery: compute RHS Poisson equation based on FOM (0) or ROM (1)
    subtract_pressure_mean::Bool = false                     # Subtract pressure mean from snapshots
    process_iteration_FOM::Bool = true                       # Compute divergence = residuals = kinetic energy etc. on FOM level
    basis_type::String = "default"                           # "default" (code chooses), "svd", "direct", "snapshot"
end

# Immersed boundary method
Base.@kwdef mutable struct IBM
    use_ibm::Bool = false                                    # Use immersed boundary method
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

# Visualization settings
Base.@kwdef mutable struct Visualization
    plotgrid::Bool = false                                   # Plot gridlines and pressure points
    do_rtp::Bool = true                                      # Do real time plotting
    rtp_type::String = "velocity"                            # "velocity", "quiver", "vorticity" or "pressure"
    rtp_n::Int = 10                                          # Number of iterations between real time plots
    initialize_processor::Function = (args...; kwargs...) -> nothing
    process!::Function = (args...; kwargs...) -> nothing
end

# Setup
Base.@kwdef struct Setup{T, N}
    case::Case = Case()
    fluid::Fluid{T} = Fluid{T}()
    model::AbstractViscosityModel{T} = LaminarModel{T}()
    grid::Grid{T, N} = Grid{T, N}()
    discretization::Operators{T} = Operators{T}()
    force::AbstractBodyForce{T} = SteadyBodyForce{T}()
    rom::ROM = ROM()
    ibm::IBM = IBM()
    time::Time{T} = Time{T}()
    solver_settings::SolverSettings{T} = SolverSettings{T}()
    visualization::Visualization = Visualization()
    bc::BC{T} = BC{T}()
end
