if isdefined(@__MODULE__, :LanguageServer)                                       #src
    include("../src/IncompressibleNavierStokes.jl")                              #src
    using .IncompressibleNavierStokes                                            #src
end                                                                              #src
using IncompressibleNavierStokes
if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

##
T = Float64
viscosity_model = LaminarModel{T}(; Re = 1000)
convection_model = NoRegConvectionModel{T}()

Nx = 60                          # Number of x-volumes
Ny = 20                          # Number of y-volumes
grid = create_grid(
    T,
    Nx,
    Ny;
    xlims = (0, 3),               # Horizontal limits (left, right)
    ylims = (-0.5, 0.5),          # Vertical limits (bottom, top)
    stretch = (1, 1),             # Stretch factor (sx, sy)
    order4 = false,               # Use 4th order in space
)

solver_settings = SolverSettings{T}(;
    pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
    p_add_solve = true,                             # Additional pressure solve for second order pressure
    abstol = 1e-10,                                 # Absolute accuracy
    reltol = 1e-14,                                 # Relative accuracy
    maxiter = 10,                                   # Maximum number of iterations
    newton_type = :approximate,
)

u_bc(x, y, t, setup) = -0.25 ≤ y ≤ 0.25 ? 1.0 - 16y^2 : 0.0
v_bc(x, y, t, setup) = zero(x)
bc = create_boundary_conditions(
    T,
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:periodic, :periodic), y = (:dirichlet, :dirichlet)),
        v = (; x = (:periodic, :periodic), y = (:dirichlet, :dirichlet)),
    ),
)

bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

setup = Setup{T,2}(; viscosity_model, convection_model, grid, force, solver_settings, bc)

build_operators!(setup);

t_start, t_end = tlims = (0.0, 10.0)

initial_velocity_u(x, y) = -0.25 ≤ y ≤ 0.25 ? 1.0 - 16y^2 : 0.0
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
);

##
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem);

##
plot_velocity(setup, V, tlims[2])

plot_vorticity(setup, V, tlims[2])
