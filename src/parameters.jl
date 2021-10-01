# Case information
Base.@kwdef mutable struct Case
    name::String = "example"                 # Case name
    is_steady::Bool = false                  # Is steady steate (not unsteady)
    visc::String = "laminar"                 # "laminar", "keps", "ML", "LES, "qr"
    regularization::String = "no"            # convective term regularization: "no", "leray", "C2", "C4"
    ibm::Bool = false                        # Use immersed boundary method
    force_unsteady::Bool = false             # false: steady forcing or no forcing; true: unsteady forcing
    initial_velocity_u::Function = () -> error("initial_velocity_u not implemented")
    initial_velocity_v::Function = () -> error("initial_velocity_v not implemented")
    initial_pressure::Function = () -> error("initial_pressure not implemented")
end

# Physical properties
Base.@kwdef mutable struct Fluid{T}
    Re::T = 1                                # Reynolds number
    U1::T = 1                                # velocity scales
    U2::T = 1                                # velocity scales
    d_layer::T = 1                           # thickness of layer
end

# Turbulent flow settings
Base.@kwdef mutable struct Visc{T}
    lm::T = 1                                # mixing length
    Cs::T = 0.17                             # Smagorinsky constant
end

# Grid parameters
Base.@kwdef mutable struct Grid{T}
    Nx::Int = 10                             # Number of volumes in the x-direction
    Ny::Int = 10                             # Number of volumes in the y-direction
    x1::T = 0                                # Left
    x2::T = 1                                # Right
    y1::T = 0                                # Bottom
    y2::T = 1                                # Top
    deltax::T = 0.1                          # Mesh sizee in x-direction
    deltay::T = 0.1                          # Mesh sizee in y-direction
    sx::T = 1                                # Stretch factor in x-direction
    sy::T = 1                                # Stretch factor in y-direction
    create_mesh::Function = () -> error("mesh not implemented")

    # Fill in later
    x::Vector{T} = T[]                       # Vector of x-points
    y::Vector{T} = T[]                       # Vector of y-points
    xp::Vector{T} = T[]
    yp::Vector{T} = T[]
    hx::Vector{T} = T[]
    hy::Vector{T} = T[]
    gx::Vector{T} = T[]
    gy::Vector{T} = T[]

    # Number of pressure points in each dimension
    Npx::Int = 0
    Npy::Int = 0

    Nux_in::Int = 0
    Nux_b::Int = 0
    Nux_t::Int = 0
    Nuy_in::Int = 0
    Nuy_b::Int = 0
    Nuy_t::Int = 0
    Nvx_in::Int = 0
    Nvx_b::Int = 0
    Nvx_t::Int = 0
    Nvy_in::Int = 0
    Nvy_b::Int = 0
    Nvy_t::Int = 0

    # Number of points in solution vector
    Nu::Int = 0
    Nv::Int = 0
    NV::Int = 0
    Np::Int = 0
    Ntot::Int = 0

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
    Ω⁻¹::Vector{T} = T[]
    Ωu⁻¹::Vector{T} = T[]
    Ωv⁻¹::Vector{T} = T[]
    Ωux::Vector{T} = T[]
    Ωvx::Vector{T} = T[]
    Ωuy::Vector{T} = T[]
    Ωvy::Vector{T} = T[]
    Ωvort::Vector{T} = T[]

    hxi::Vector{T} = T[]
    hyi::Vector{T} = T[]
    hxd::Vector{T} = T[]
    hyd::Vector{T} = T[]
    gxi::Vector{T} = T[]
    gyi::Vector{T} = T[]
    gxd::Vector{T} = T[]
    gyd::Vector{T} = T[]

    Buvy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Bvux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Bkux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Bkvy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    xin::Vector{T} = T[]
    yin::Vector{T} = T[]

    # Separate grids for u, v, and p
    xu::Matrix{T} = zeros(T, 0, 0)
    yu::Matrix{T} = zeros(T, 0, 0)
    xv::Matrix{T} = zeros(T, 0, 0)
    yv::Matrix{T} = zeros(T, 0, 0)
    xpp::Matrix{T} = zeros(T, 0, 0)
    ypp::Matrix{T} = zeros(T, 0, 0)

    # Ranges
    indu::UnitRange{Int} = 1:0
    indv::UnitRange{Int} = 1:0
    indV::UnitRange{Int} = 1:0
    indp::UnitRange{Int} = 1:0
end


# Discretization parameters
Base.@kwdef mutable struct Discretization{T}
    order4::Bool = false                     # Use 4th order in space (otherwise 2nd order)
    α::T = 81                                # richardson extrapolation factor = 3^4
    β::T = 9 // 8                            # interpolation factor

    # Filled in by function
    Au_ux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Au_uy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Av_vx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Av_vy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Au_ux_bc::NamedTuple = (;)
    Au_uy_bc::NamedTuple = (;)
    Av_vx_bc::NamedTuple = (;)
    Av_vy_bc::NamedTuple = (;)

    Iu_ux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Iv_uy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Iu_vx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Iv_vy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Iu_ux_bc::NamedTuple = (;)
    Iv_vy_bc::NamedTuple = (;)
    Iv_uy_bc_lr::NamedTuple = (;)
    Iv_uy_bc_lu::NamedTuple = (;)
    Iu_vx_bc_lr::NamedTuple = (;)
    Iu_vx_bc_lu::NamedTuple = (;)

    M::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Mx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    My::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Mx_bc::NamedTuple = (;)
    My_bc::NamedTuple = (;)

    G::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Gx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Gy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Bup::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Bvp::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Cux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Cuy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Cvx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Cvy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Su_ux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Su_uy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Su_vx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    # Su_vy never defined
    # Sv_ux never defined
    Sv_uy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Sv_vx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Sv_vy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Su_ux_bc::NamedTuple = (;)
    Su_uy_bc::NamedTuple = (;)
    Sv_vx_bc::NamedTuple = (;)
    Sv_vy_bc::NamedTuple = (;)

    Su_vx_bc_lr::NamedTuple = (;)
    Su_vx_bc_lu::NamedTuple = (;)
    Sv_uy_bc_lr::NamedTuple = (;)
    Sv_uy_bc_lu::NamedTuple = (;)

    Dux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Duy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Dvx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Dvy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Diffu::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Diffv::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    Wv_vx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Wu_uy::SparseMatrixCSC{T,Int} = spzeros(0, 0)

    yM::Vector{T} = T[]
    y_px::Vector{T} = T[]
    y_py::Vector{T} = T[]
    yAu_ux::Vector{T} = T[]
    yAu_uy::Vector{T} = T[]
    yAv_vx::Vector{T} = T[]
    yAv_vy::Vector{T} = T[]
    yDiffu::Vector{T} = T[]
    yDiffv::Vector{T} = T[]
    yIu_ux::Vector{T} = T[]
    yIv_uy::Vector{T} = T[]
    yIu_vx::Vector{T} = T[]
    yIv_vy::Vector{T} = T[]

    ySu_ux::Vector{T} = T[] # TODO: Vectors or matrices?
    ySu_uy::Vector{T} = T[] # TODO: Vectors or matrices?
    ySu_vx::Vector{T} = T[] # TODO: Vectors or matrices?
    ySv_vx::Vector{T} = T[] # TODO: Vectors or matrices?
    ySv_vy::Vector{T} = T[] # TODO: Vectors or matrices?
    ySv_uy::Vector{T} = T[] # TODO: Vectors or matrices?

    Anu_vy_bc::NamedTuple = (;)

    Cux_k_bc::NamedTuple = (;)
    Cuy_k_bc::NamedTuple = (;)
    Cvx_k_bc::NamedTuple = (;)
    Cvy_k_bc::NamedTuple = (;)

    Auy_k_bc::NamedTuple = (;)
    Avx_k_bc::NamedTuple = (;)

    A::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    A_fact::Factorization{T} = cholesky(spzeros(0, 0))

    lu_diffu::Factorization{T} = cholesky(spzeros(0, 0))
    lu_diffv::Factorization{T} = cholesky(spzeros(0, 0))

    ydM::Vector{T} = T[]
    ypx::Vector{T} = T[]
    ypy::Vector{T} = T[]

    Aν_ux::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Aν_uy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Aν_vx::SparseMatrixCSC{T,Int} = spzeros(0, 0)
    Aν_vy::SparseMatrixCSC{T,Int} = spzeros(0, 0)
end

# Forcing parameters
Base.@kwdef mutable struct Force{T}
    x_c::T = 0                               # x-coordinate of body
    y_c::T = 0                               # y-coordinate of body
    Ct::T = 0                                # thrust coefficient for actuator disk computations
    D::T = 1                                 # Actuator disk diameter
    Fx::Vector{T} = T[]                      # For storing constant body force
    Fy::Vector{T} = T[]                      # For storing constant body force
    isforce::Bool = false                    # presence of a force file
    force_unsteady::Bool = false             # is_steady (false) or unsteady (true) force
    bodyforce_x::Function = () -> error("bodyforce_x not implemented")
    bodyforce_y::Function = () -> error("bodyforce_y not implemented")
end

# Rom parameters
Base.@kwdef mutable struct ROM
    use_rom::Bool = false                    # use reduced order model
    rom_type::String = "POD"                 # "POD",  "Fourier"
    M::Int = 10                              # number of velocity modes for reduced order model
    Mp::Int = 10                             # number of pressure modes for reduced order model
    precompute_convection::Bool = true       # precomputed convection matrices
    precompute_diffusion::Bool = true        # precomputed diffusion matrices
    precompute_force::Bool = true            # precomputed forcing term
    t_snapshots::Int = 0                     # snapshots
    Δt_snapshots::Bool = false
    mom_cons::Bool = false                   # momentum conserving SVD
    rom_bc::Int = 0                          # 0: homogeneous (no-slip = periodic) 1: non-homogeneous = time-independent 2: non-homogeneous = time-dependent
    weighted_norm::Bool = true               # Use weighted norm (using finite volumes as weights)
    pressure_recovery::Bool = false          # false: no pressure computation, true: compute pressure with PPE-ROM
    pressure_precompute::Int = 0             # in case of pressure_recovery: compute RHS Poisson equation based on FOM (0) or ROM (1)
    subtract_pressure_mean::Bool = false     # Subtract pressure mean from snapshots
    process_iteration_FOM::Bool = true       # compute divergence = residuals = kinetic energy etc. on FOM level
    basis_type::String = "default"           # "default" (code chooses), "svd", "direct", "snapshot"
end

# Immersed boundary method
Base.@kwdef mutable struct IBM
    ibm::Bool = false                        # use immersed boundary method
end

# Time stepping
Base.@kwdef mutable struct Time{T}
    t_start::T = 0                           # Start time
    t_end::T = 1                             # End time
    Δt::T = (t_end - t_start) / 100          # Timestep
    rk_method::RungeKuttaMethod = RK44()     # Runge Kutta method
    method::Int = 0                          # Method number
    method_startup::Int = 0                  # Startup method for methods that are not self-starting
    method_startup_number::Int = 0           # number of velocity fields necessary for start-up = equal to order of method
    isadaptive::Bool = false                 # Adapt timestep every n_adapt_Δt iterations
    n_adapt_Δt::Int = 1                      # Number of iterations between timestep adjustment
    θ::T = 1 // 2                            # θ value for implicit θ method
    β::T = 1 // 2                            # β value for oneleg β method
    CFL::T = 1 // 2                          # CFL number for adaptive methods
end

# Solver settings
Base.@kwdef mutable struct SolverSettings{T}
    p_initial::Bool = true                   # calculate compatible IC for the pressure
    p_add_solve::Bool = true                 # additional pressure solve to make it same order as velocity

    # Accuracy for non-linear solves (method 62 = 72 = 9)
    nonlinear_acc::T = 1e-14                 # Absolute accuracy
    nonlinear_relacc::T = 1e-14              # Relative accuracy
    nonlinear_maxit::Int = 10                # Maximum number of iterations

    # "no": do not compute Jacobian, but approximate iteration matrix with I/Δt
    # "approximate: approximate Newton build Jacobian once at beginning of nonlinear iterations
    # "full": full Newton build Jacobian at each iteration
    nonlinear_Newton::String = "approximate"

    Jacobian_type::String = "picard"         # "picard": Picard linearization, "newton": Newton linearization
    Newton_factor::Bool = false              # Newton factor
    nonlinear_startingvalues::Bool = false   # Extrapolate values from last time step to get accurate initial guess (for unsteady problems only)
    nPicard::Int = 5                         # number of Picard steps before switching to Newton when linearization is Newton (for is_steady problems only)
end

# Output files
Base.@kwdef mutable struct Output
    save_results::Bool = false               # Save results
    savepath::String = "results"             # Path for saving
    save_unsteady::Bool = false              # Save intermediate time steps
end

# Visualization settings
Base.@kwdef mutable struct Visualization
    plotgrid::Bool = false                   # Plot gridlines and pressure points
    do_rtp::Bool = true                      # Do real time plotting
    rtp_type::String = "velocity"            # "velocity", "quiver", "vorticity" or "pressure"
    rtp_n::Int = 10                          # Number of iterations between real time plots
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

Base.@kwdef struct Setup{T}
    case::Case = Case()
    fluid::Fluid{T} = Fluid{T}()
    visc::Visc{T} = Visc{T}()
    grid::Grid{T} = Grid{T}()
    discretization::Discretization{T} = Discretization{T}()
    force::Force{T} = Force{T}()
    rom::ROM = ROM()
    ibm::IBM = IBM()
    time::Time{T} = Time{T}()
    solver_settings::SolverSettings{T} = SolverSettings{T}()
    output::Output = Output()
    visualization::Visualization = Visualization()
    bc::BC{T} = BC{T}()
end
