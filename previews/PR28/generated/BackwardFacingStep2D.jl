using IncompressibleNavierStokes

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

name = "BackwardFacingStep2D"

T = Float64

viscosity_model = LaminarModel{T}(; Re = 3000)
# viscosity_model = KEpsilonModel{T}(; Re = 2000)
# viscosity_model = MixingLengthModel{T}(; Re = 2000)
# viscosity_model = SmagorinskyModel{T}(; Re = 2000)
# viscosity_model = QRModel{T}(; Re = 2000)

convection_model = NoRegConvectionModel()
# convection_model = C2ConvectionModel()
# convection_model = C4ConvectionModel()
# convection_model = LerayConvectionModel()

u_bc(x, y, t) = x ≈ 0 && y ≥ 0 ? 24y * (1 / 2 - y) : 0.0
v_bc(x, y, t) = 0.0
boundary_conditions = BoundaryConditions(
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:dirichlet, :pressure), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :symmetric), y = (:dirichlet, :dirichlet)),
    ),
    T,
)

x = stretched_grid(0.0, 10.0, 300)
y = cosine_grid(-0.5, 0.5, 50)
grid = Grid(x, y; boundary_conditions, T);

plot_grid(grid)

bodyforce_u(x, y) = 0.0
bodyforce_v(x, y) = 0.0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

setup = Setup(; viscosity_model, convection_model, grid, force, boundary_conditions)

pressure_solver = DirectPressureSolver(setup)

t_start, t_end = tlims = (0.0, 7.0)

initial_velocity_u(x, y) = y ≥ 0.0 ? 24y * (1 / 2 - y) : 0.0
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
    pressure_solver,
);

problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem);

logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 5, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 20, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 10)
# processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); inplace = true, Δt = 0.002, processors, pressure_solver);

plot_tracers(tracer)

plot_pressure(setup, p)

plot_velocity(setup, V, t_end)

plot_vorticity(setup, V, tlims[2])

plot_streamfunction(setup, V, tlims[2])

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

