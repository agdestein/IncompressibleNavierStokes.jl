using IncompressibleNavierStokes

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

T = Float64

viscosity_model = LaminarModel{T}(; Re = 1000)

convection_model = NoRegConvectionModel()

u_bc(x, y, t) = y ≈ 1 ? 1.0 : 0.0
v_bc(x, y, t) = zero(x)
boundary_conditions = BoundaryConditions(
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    ),
    T,
)

x = cosine_grid(0.0, 1.0, 50)
y = stretched_grid(0.0, 1.0, 50, 0.95)
grid = Grid(x, y; boundary_conditions, T)

plot_grid(grid)

bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

setup = Setup(; viscosity_model, convection_model, grid, force, boundary_conditions)

pressure_solver = DirectPressureSolver(setup)

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
    pressure_solver,
)

problem = SteadyStateProblem(setup, V₀, p₀)
V, p = @time solve(problem)

problem = UnsteadyProblem(setup, V₀, p₀, tlims)

logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 50, fieldname = :vorticity, type = contourf)
writer = VTKWriter(; nupdate = 20, dir = "output/LidDrivenCavity2D")
tracer = QuantityTracer(; nupdate = 10)
# processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

V, p = @time solve(problem, RK44(); Δt = 0.001, processors, pressure_solver)

plot_tracers(tracer)

plot_pressure(setup, p)

plot_velocity(setup, V, t_end)

levels = [-7, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 7]
plot_vorticity(setup, V, tlims[2]; levels)

plot_streamfunction(setup, V, tlims[2])

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

