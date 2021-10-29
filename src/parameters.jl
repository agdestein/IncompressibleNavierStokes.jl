# Case information
Base.@kwdef mutable struct Case
    name::String = "example"                 # Case name
    problem::Problem = UnsteadyProblem()     # Problem type
    regularization::String = "no"            # Convective term regularization: "no", "leray", "C2", "C4"
    ibm::Bool = false                        # Use immersed boundary method
    force_unsteady::Bool = false             # False: steady forcing or no forcing; true: unsteady forcing
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

# Grid parameters
Base.@kwdef mutable struct Grid{T, N}
    Nx::Int = 10                             # Number of x-volumes
    Ny::Int = 10                             # Number of y-volumes
    Nz::Int = 0                              # Number of z-volumes (if any)
    xlims::Tuple{T,T} = (0, 1)               # Horizontal limits (left, right)
    ylims::Tuple{T,T} = (0, 1)               # Vertical limits (bottom, top)
    stretch::NTuple{N,T} = (1, 1)            # Stretch factor (sx, sy[, sz])
    create_mesh::Function = () -> error("mesh not implemented")

    # Fill in later
    Δx::T = 0                                # Mesh size in x-direction
    Δy::T = 0                                # Mesh size in y-direction
    Δz::T = 0                                # Mesh size in z-direction
    x::Vector{T} = T[]                       # Vector of x-points
    y::Vector{T} = T[]                       # Vector of y-points
    z::Vector{T} = T[]                       # Vector of z-points
    xp::Vector{T} = T[]
    yp::Vector{T} = T[]
    zp::Vector{T} = T[]

    # Number of pressure points in each dimension
    Npx::Int = 0
    Npy::Int = 0
    Npz::Int = 0

    Nux_in::Int = 0
    Nux_b::Int = 0
    Nux_t::Int = 0
    Nuy_in::Int = 0
    Nuy_b::Int = 0
    Nuy_t::Int = 0
    Nuz_in::Int = 0
    Nuz_b::Int = 0
    Nuz_t::Int = 0
    
    Nvx_in::Int = 0
    Nvx_b::Int = 0
    Nvx_t::Int = 0
    Nvy_in::Int = 0
    Nvy_b::Int = 0
    Nvy_t::Int = 0
    Nvz_in::Int = 0
    Nvz_b::Int = 0
    Nvz_t::Int = 0

    Nwx_in::Int = 0
    Nwx_b::Int = 0
    Nwx_t::Int = 0
    Nwy_in::Int = 0
    Nwy_b::Int = 0
    Nwy_t::Int = 0
    Nwz_in::Int = 0
    Nwz_b::Int = 0
    Nwz_t::Int = 0

    # Number of points in solution vector
    Nu::Int = 0
    Nv::Int = 0
    Nw::Int = 0
    NV::Int = 0
    Np::Int = 0

    N1::Int = 0
    N2::Int = 0
    N3::Int = 0
    N4::Int = 0

    # Operator mesh?
    Ωp::Vector{T} = T[]
    Ωp⁻¹::Vector{T} = T[]
    Ω::Vector{T} = T[]
    Ωu::Vector{T} = T[]
    Ωv::Vector{T} = T[]
    Ωw::Vector{T} = T[]
    Ω⁻¹::Vector{T} = T[]
    Ωu⁻¹::Vector{T} = T[]
    Ωv⁻¹::Vector{T} = T[]
    Ωw⁻¹::Vector{T} = T[]
    Ωux::Vector{T} = T[]
    Ωuy::Vector{T} = T[]
    Ωuz::Vector{T} = T[]
    Ωvx::Vector{T} = T[]
    Ωvy::Vector{T} = T[]
    Ωvz::Vector{T} = T[]
    Ωwx::Vector{T} = T[]
    Ωwy::Vector{T} = T[]
    Ωwz::Vector{T} = T[]
    Ωvort::Vector{T} = T[]

    hx::Vector{T} = T[]
    hy::Vector{T} = T[]
    hz::Vector{T} = T[]
    hxi::Vector{T} = T[]
    hyi::Vector{T} = T[]
    hzi::Vector{T} = T[]
    hxd::Vector{T} = T[]
    hyd::Vector{T} = T[]
    hzd::Vector{T} = T[]
    gx::Vector{T} = T[]
    gy::Vector{T} = T[]
    gz::Vector{T} = T[]
    gxi::Vector{T} = T[]
    gyi::Vector{T} = T[]
    gzi::Vector{T} = T[]
    gxd::Vector{T} = T[]
    gyd::Vector{T} = T[]
    gzd::Vector{T} = T[]

    Buvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bkux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bkvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    xin::Vector{T} = T[]
    yin::Vector{T} = T[]
    zin::Vector{T} = T[]

    # Separate grids for u, v, and p
    xu::Matrix{T} = zeros(T, 0, 0)
    xv::Matrix{T} = zeros(T, 0, 0)
    xw::Matrix{T} = zeros(T, 0, 0)
    yu::Matrix{T} = zeros(T, 0, 0)
    yv::Matrix{T} = zeros(T, 0, 0)
    yw::Matrix{T} = zeros(T, 0, 0)
    zu::Matrix{T} = zeros(T, 0, 0)
    zv::Matrix{T} = zeros(T, 0, 0)
    zw::Matrix{T} = zeros(T, 0, 0)
    xpp::Matrix{T} = zeros(T, 0, 0)
    ypp::Matrix{T} = zeros(T, 0, 0)
    zpp::Matrix{T} = zeros(T, 0, 0)

    # Ranges
    indu::UnitRange{Int} = 0:0
    indv::UnitRange{Int} = 0:0
    indw::UnitRange{Int} = 0:0
    indV::UnitRange{Int} = 0:0
    indp::UnitRange{Int} = 0:0
end


# Discretization parameters
Base.@kwdef mutable struct Discretization{T}
    order4::Bool = false                     # Use 4th order in space (otherwise 2nd order)
    α::T = 81                                # Richardson extrapolation factor = 3^4
    β::T = 9 // 8                            # Interpolation factor

    # Filled in by function
    Au_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Au_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Au_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Au_ux_bc::NamedTuple = (;)
    Au_uy_bc::NamedTuple = (;)
    Au_uz_bc::NamedTuple = (;)
    Av_vx_bc::NamedTuple = (;)
    Av_vy_bc::NamedTuple = (;)
    Av_vz_bc::NamedTuple = (;)
    Aw_wx_bc::NamedTuple = (;)
    Aw_wy_bc::NamedTuple = (;)
    Aw_wz_bc::NamedTuple = (;)

    Iu_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iv_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iu_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iv_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Iu_ux_bc::NamedTuple = (;)
    Iv_vy_bc::NamedTuple = (;)
    Iv_uy_bc_lr::NamedTuple = (;)
    Iv_uy_bc_lu::NamedTuple = (;)
    Iu_vx_bc_lr::NamedTuple = (;)
    Iu_vx_bc_lu::NamedTuple = (;)

    M::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Mx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    My::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Mx_bc::NamedTuple = (;)
    My_bc::NamedTuple = (;)

    G::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Gx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Gy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Bup::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvp::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Cux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cuy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Su_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    # Su_vy never defined
    # Sv_ux never defined
    Sv_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Su_ux_bc::NamedTuple = (;)
    Su_uy_bc::NamedTuple = (;)
    Sv_vx_bc::NamedTuple = (;)
    Sv_vy_bc::NamedTuple = (;)

    Su_vx_bc_lr::NamedTuple = (;)
    Su_vx_bc_lu::NamedTuple = (;)
    Sv_uy_bc_lr::NamedTuple = (;)
    Sv_uy_bc_lu::NamedTuple = (;)

    Dux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Duy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dvx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Diff::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Wv_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Wu_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    yM::Vector{T} = T[]
    y_p::Vector{T} = T[]
    yAu_ux::Vector{T} = T[]
    yAu_uy::Vector{T} = T[]
    yAv_vx::Vector{T} = T[]
    yAv_vy::Vector{T} = T[]
    yDiff::Vector{T} = T[]
    yIu_ux::Vector{T} = T[]
    yIv_uy::Vector{T} = T[]
    yIu_vx::Vector{T} = T[]
    yIv_vy::Vector{T} = T[]

    ySu_ux::Vector{T} = T[]
    ySu_uy::Vector{T} = T[]
    ySu_vx::Vector{T} = T[]
    ySv_vx::Vector{T} = T[]
    ySv_vy::Vector{T} = T[]
    ySv_uy::Vector{T} = T[]

    Cux_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cuy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvx_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Cux_k_bc::NamedTuple = (;)
    Cuy_k_bc::NamedTuple = (;)
    Cvx_k_bc::NamedTuple = (;)
    Cvy_k_bc::NamedTuple = (;)

    Auy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Avx_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Auy_k_bc::NamedTuple = (;)
    Avx_k_bc::NamedTuple = (;)

    A::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    A_fact::Factorization{T} = cholesky(spzeros(T, 0, 0))

    ydM::Vector{T} = T[]
    ypx::Vector{T} = T[]
    ypy::Vector{T} = T[]

    Aν_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_ux_bc::NamedTuple = (;)
    Aν_vy_bc::NamedTuple = (;)
    Aν_uy_bc_lr::NamedTuple = (;)
    Aν_uy_bc_lu::NamedTuple = (;)
    Aν_vx_bc_lr::NamedTuple = (;)
    Aν_vx_bc_lu::NamedTuple = (;)

    yAν_ux::Vector{T} = T[]
    yAν_uy::Vector{T} = T[]
    yAν_vx::Vector{T} = T[]
    yAν_vy::Vector{T} = T[]

    yCux_k::Vector{T} = T[]
    yCuy_k::Vector{T} = T[]
    yCvx_k::Vector{T} = T[]
    yCvy_k::Vector{T} = T[]
    yAuy_k::Vector{T} = T[]
    yAvx_k::Vector{T} = T[]
end

# Forcing parameters
Base.@kwdef mutable struct Force{T}
    x_c::T = 0                                               # X-coordinate of body
    y_c::T = 0                                               # Y-coordinate of body
    Ct::T = 0                                                # Thrust coefficient for actuator disk computations
    D::T = 1                                                 # Actuator disk diameter
    F::Vector{T} = T[]                                       # For storing constant body force
    isforce::Bool = false                                    # Presence of a force file
    force_unsteady::Bool = false                             # Is_steady (false) or unsteady (true) force
    bodyforce_x::Function = () -> error("bodyforce_x not implemented")
    bodyforce_y::Function = () -> error("bodyforce_y not implemented")
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
    Â::Matrix{T} = zeros(T, 0, 0)                            # Fourier transform of `A`

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

# Boundary conditions
Base.@kwdef mutable struct BC{T}
    bc_unsteady::Bool = false
    u::NamedTuple = (;)
    v::NamedTuple = (;)
    k::NamedTuple = (;)
    e::NamedTuple = (;)
    ν::NamedTuple = (;)
    u_bc::Function = () -> error("u_bc not implemented")
    v_bc::Function = () -> error("v_bc not implemented")
    dudt_bc::Function = () -> error("dudt_bc not implemented")
    dvdt_bc::Function = () -> error("dvdt_bc not implemented")
    pLe::T = 0
    pRi::T = 0
    pLo::T = 0
    pUp::T = 0
    bc_type::Function = () -> error("bc_type not implemented")
end

Base.@kwdef struct Setup{T, N}
    case::Case = Case()
    fluid::Fluid{T} = Fluid{T}()
    model::ViscosityModel{T} = LaminarModel{T}()
    grid::Grid{T, N} = Grid{T, N}()
    discretization::Discretization{T} = Discretization{T}()
    force::Force{T} = Force{T}()
    rom::ROM = ROM()
    ibm::IBM = IBM()
    time::Time{T} = Time{T}()
    solver_settings::SolverSettings{T} = SolverSettings{T}()
    visualization::Visualization = Visualization()
    bc::BC{T} = BC{T}()
end
