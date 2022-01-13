using IncompressibleNavierStokes

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

T = Float64

viscosity_model = LaminarModel{T}(; Re = 1000)

convection_model = NoRegConvectionModel{T}()

Nx = 100                          # Number of x-volumes
Ny = 100                          # Number of y-volumes
grid = create_grid(
    T,
    Nx,
    Ny;
    xlims = (0, 1),               # Horizontal limits (left, right)
    ylims = (0, 1),               # Vertical limits (bottom, top)
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

u_bc(x, y, t, setup) = y ≈ setup.grid.ylims[2] ? 1.0 : 0.0
v_bc(x, y, t, setup) = zero(x)
bc = create_boundary_conditions(
    T,
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        k = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        e = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ν = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    ),
)

bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

setup = Setup{T,2}(; viscosity_model, convection_model, grid, force, solver_settings, bc)

build_operators!(setup);

t_start, t_end = tlims = (0.0, 10.0)

initial_velocity_u(x, y) = 0
initial_velocity_v(x, y) = 0
initial_pressure(x, y) = 0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
);

logger = Logger(; nupdate = 10)
real_time_plotter = RealTimePlotter(; nupdate = 5, fieldname = :vorticity)
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, real_time_plotter, tracer]

problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem; processors);

problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.01, processors);

plot_tracers(tracer)

plot_pressure(setup, p)

plot_velocity(setup, V, t_end)

plot_vorticity(setup, V, tlims[2])

plot_streamfunction(setup, V, tlims[2])

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

