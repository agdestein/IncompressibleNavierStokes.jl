# Case information
Base.@kwdef mutable struct Case
    name::String = "example"                 # Case name
    steady::Bool = false                     # Is steady steate (not unsteady)
    visc::String = "laminar"                 # 'laminar', 'keps', 'ML', 'LES, 'qr'
    regularization::String = "no"            # convective term regularization: "no", "leray", "C2", "C4"
    ibm::Bool = false                        # Use immersed boundary method
    force_unsteady::Bool = false             # false: steady forcing or no forcing; true: unsteady forcing
    initial_velocity_u::Function = () -> error("initial_velocity_u function not implemented")
    initial_velocity_v::Function = () -> error("initial_velocity_v function not implemented")
    initial_pressure::Function = () -> error("initial_pressure function not implemented")
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
    create_mesh::Function = () -> error("mesh function not implemented")

    # Fill in later
    x::Vector{T} = T[]                       # Vector of x-points
    y::Vector{T} = T[]                       # Vector of y-points
    xp::Vector{T} = T[]
    yp::Vector{T} = T[]
    hx::Vector{T} = T[]
    hy::Vector{T} = T[]
    gx::Vector{T} = T[]
    gy::Vector{T} = T[]
    Npx::Int = 0
    Npy::Int = 0
    Np::Int = 0
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
    Nu::Int = 0
    Nv::Int = 0
    NV::Int = 0
    Ntot::Int = 0
    N1::Int = 0
    N2::Int = 0
    N3::Int = 0
    N4::Int = 0
    Omp::Vector{T} = T[]
    Omp_inv::Vector{T} = T[]
    Om::Vector{T} = T[]
    Omu::Vector{T} = T[]
    Omv::Vector{T} = T[]
    Om_inv::Vector{T} = T[]
    Omu_inv::Vector{T} = T[]
    Omv_inv::Vector{T} = T[]
    Omux::Vector{T} = T[]
    Omvx::Vector{T} = T[]
    Omuy::Vector{T} = T[]
    Omvy::Vector{T} = T[]
    Omvort::Vector{T} = T[]
    hxi::Vector{T} = T[]
    hyi::Vector{T} = T[]
    hxd::Vector{T} = T[]
    hyd::Vector{T} = T[]
    gxi::Vector{T} = T[]
    gyi::Vector{T} = T[]
    gxd::Vector{T} = T[]
    gyd::Vector{T} = T[]
    Buvy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Bvux::Matrix{T} = Matrix{T}(undef, 0, 0)
    Bkux::Matrix{T} = Matrix{T}(undef, 0, 0)
    Bkvy::Matrix{T} = Matrix{T}(undef, 0, 0)
    xin::Vector{T} = T[]
    yin::Vector{T} = T[]
    xu::Matrix{T} = Matrix{T}(undef, 0, 0)
    yu::Matrix{T} = Matrix{T}(undef, 0, 0)
    xv::Matrix{T} = Matrix{T}(undef, 0, 0)
    yv::Matrix{T} = Matrix{T}(undef, 0, 0)
    xpp::Matrix{T} = Matrix{T}(undef, 0, 0)
    ypp::Matrix{T} = Matrix{T}(undef, 0, 0)
    indu::Vector{T} = T[]
    indv::Vector{T} = T[]
    indV::Vector{T} = T[]
    indp::Vector{T} = T[]
end


# Discretization parameters
Base.@kwdef mutable struct Discretization{T}
    order4::Bool = false                     # Use 4th order in time (otherwise 2nd order)
    α::T = 81                                # richardson extrapolation factor = 3^4
    β::T = 9 / 8                            # interpolation factor

    # Filled in by function
    Au_ux::Matrix{T} = Matrix{T}(undef, 0, 0)
    Au_uy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Av_vx::Matrix{T} = Matrix{T}(undef, 0, 0)
    Av_vy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Au_ux_bc::NamedTuple = (;)
    Au_uy_bc::NamedTuple = (;)
    Av_vx_bc::NamedTuple = (;)
    Av_vy_bc::NamedTuple = (;)
    Iu_ux::Matrix{T} = Matrix{T}(undef, 0, 0)
    Iv_uy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Iu_vx::Matrix{T} = Matrix{T}(undef, 0, 0)
    Iv_vy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Iu_ux_bc::NamedTuple = (;)
    Iv_vy_bc::NamedTuple = (;)
    Iv_uy_bc_lr::NamedTuple = (;)
    Iv_uy_bc_lu::NamedTuple = (;)
    Iu_vx_bc_lr::NamedTuple = (;)
    Iu_vx_bc_lu::NamedTuple = (;)
    M::Matrix{T} = Matrix{T}(undef, 0, 0)
    Mx::Matrix{T} = Matrix{T}(undef, 0, 0)
    My::Matrix{T} = Matrix{T}(undef, 0, 0)
    Mx_bc::NamedTuple = (;)
    My_bc::NamedTuple = (;)
    G::Matrix{T} = Matrix{T}(undef, 0, 0)
    Gx::Matrix{T} = Matrix{T}(undef, 0, 0)
    Gy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Bup::Matrix{T} = Matrix{T}(undef, 0, 0)
    Bvp::Matrix{T} = Matrix{T}(undef, 0, 0)
    Cux::Matrix{T} = Matrix{T}(undef, 0, 0)
    Cuy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Cvx::Matrix{T} = Matrix{T}(undef, 0, 0)
    Cvy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Su_ux::Matrix{T} = Matrix{T}(undef, 0, 0)
    Su_uy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Sv_vx::Matrix{T} = Matrix{T}(undef, 0, 0)
    Sv_vy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Su_ux_bc::NamedTuple = (;)
    Su_uy_bc::NamedTuple = (;)
    Sv_vx_bc::NamedTuple = (;)
    Sv_vy_bc::NamedTuple = (;)
    Dux::Matrix{T} = Matrix{T}(undef, 0, 0)
    Duy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Dvx::Matrix{T} = Matrix{T}(undef, 0, 0)
    Dvy::Matrix{T} = Matrix{T}(undef, 0, 0)
    Diffu::Matrix{T} = Matrix{T}(undef, 0, 0)
    Diffv::Matrix{T} = Matrix{T}(undef, 0, 0)
    Su_vx_bc_lr::NamedTuple = (;)
    Su_vx_bc_lu::NamedTuple = (;)
    Sv_uy_bc_lr::NamedTuple = (;)
    Sv_uy_bc_lu::NamedTuple = (;)
    Wv_vx::Matrix{T} = Matrix{T}(undef, 0, 0)
    Wu_uy::Matrix{T} = Matrix{T}(undef, 0, 0)
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
end

# Forcing parameters
Base.@kwdef mutable struct Force{T}
    x_c::T = 0                               # x-coordinate of body
    y_c::T = 0                               # y-coordinate of body
    Ct::T = 0                                # thrust coefficient for actuator disk computations
    D::T = 1                                 # Actuator disk diameter
    isforce::Bool = false                    # presence of a force file
    force_unsteady::Bool = false             # steady (0) or unsteady (1) force
    bodyforce_x::Function = () -> error("bodyforce_x function not implemented")
    bodyforce_y::Function = () -> error("bodyforce_y function not implemented")
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
    dt_snapshots::Bool = false
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

# Time marching
Base.@kwdef mutable struct Time{T}
    t_start::T = 0                           # Start time
    t_end::T = 1                             # End time
    dt::T = (t_end - t_start) / 100          # Timestep
    RK::String = "RK44"                      # RK method
    method::Int = 0                          # Method number
    θ::T = 1 // 2                            # θ value for implicit θ method
    β::T = 1 // 2                         # β value for oneleg β method
end

# Solver settings
Base.@kwdef mutable struct SolverSettings{T}
    p_initial::Bool = true                   # calculate compatible IC for the pressure
    p_add_solve::Bool = true                 # additional pressure solve to make it same order as velocity

    # Accuracy for non-linear solves (method 62 = 72 = 9)
    nonlinear_acc::T = 1e-14                 # Absolute accuracy
    nonlinear_relacc::T = 1e-14              # Relative accuracy
    nonlinear_maxit::Int = 10                # Maximum number of iterations

    # "no": do not compute Jacobian, but approximate iteration matrix with I/dt
    # "approximate: approximate Newton build Jacobian once at beginning of nonlinear iterations
    # "full": full Newton build Jacobian at each iteration
    nonlinear_Newton::String = "approximate"

    Jacobian_type::String = "picard"         # "picard": Picard linearization, "newton": Newton linearization
    nonlinear_startingvalues::Bool = false   # Extrapolate values from last time step to get accurate initial guess (for unsteady problems only)
    nPicard::Int = 5                         # number of Picard steps before switching to Newton when linearization is Newton (for steady problems only)
end

# Output files
Base.@kwdef mutable struct Output
    save_results::Bool = false               # Save results
    savepath::String = "results"             # Path for saving
    save_unsteady::Bool = false              # Save intermediate time steps
end

# Visualization settings
Base.@kwdef mutable struct Visualization
    plotgrid::Bool = false                   # plot gridlines and pressure points
    rtp_show::Bool = true                    # real time plotting
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
    u_bc::Function = () -> error("u_bc function not implemented")
    v_bc::Function = () -> error("v_bc function not implemented")
    dudt_bc::Function = () -> error("dudt_bc function not implemented")
    dvdt_bc::Function = () -> error("dvdt_bc function not implemented")
    pLe::T = 0
    pRi::T = 0
    pLo::T = 0
    pUp::T = 0
    boundary_conditions::Function = () -> error("boundary_conditions function not implemented")
    bc_type::Function = () -> error("bc_type function not implemented")
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
