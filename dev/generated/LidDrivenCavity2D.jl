using IncompressibleNavierStokes

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

T = Float64

viscosity_model = LaminarModel{T}(; Re = 1000)

convection_model = NoRegConvectionModel{T}()

x = cosine_grid(0, 1, 50)
y = stretched_grid(0, 1, 50, 0.95)
grid = create_grid(x, y; T)

plot_grid(grid)

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
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    ),
    T,
)

bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

setup = Setup{T,2}(; viscosity_model, convection_model, grid, force, solver_settings, bc)

build_operators!(setup)

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
)

problem = SteadyStateProblem(setup, V₀, p₀)
V, p = @time solve(problem)

problem = UnsteadyProblem(setup, V₀, p₀, tlims)

logger = Logger(; nupdate = 20)
plotter = RealTimePlotter(; nupdate = 20, fieldname = :vorticity)
writer = VTKWriter(; nupdate = 20, dir = "output/LidDrivenCavity2D")
tracer = QuantityTracer(; nupdate = 10)
processors = [logger, plotter, writer, tracer]

V, p = @time solve(problem, RK44(); Δt = 0.001, processors)

plot_tracers(tracer)

plot_pressure(setup, p)

plot_velocity(setup, V, t_end)

plot_vorticity(setup, V, tlims[2])

plot_streamfunction(setup, V, tlims[2])

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

