using IncompressibleNavierStokes

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

name = "Actuator2D"

T = Float64

viscosity_model = LaminarModel{T}(; Re = 100)
# viscosity_model = KEpsilonModel{T}(; Re = 100)
# viscosity_model = MixingLengthModel{T}(; Re = 100)
# viscosity_model = SmagorinskyModel{T}(; Re = 100)
# viscosity_model = QRModel{T}(; Re = 100)

convection_model = NoRegConvectionModel()
# convection_model = C2ConvectionModel()
# convection_model = C4ConvectionModel()
# convection_model = LerayConvectionModel()

f = 0.5
u_bc(x, y, t) = x ≈ 0.0 ? cos(π / 6 * sin(f * t)) : 0.0
v_bc(x, y, t) = x ≈ 0.0 ? sin(π / 6 * sin(f * t)) : 0.0
dudt_bc(x, y, t) = x ≈ 0.0 ? -π / 6 * f * cos(f * t) * sin(π / 6 * sin(f * t)) : 0.0
dvdt_bc(x, y, t) = x ≈ 0.0 ? π / 6 * f * cos(f * t) * cos(π / 6 * sin(f * t)) : 0.0
boundary_conditions = BoundaryConditions(
    u_bc,
    v_bc;
    dudt_bc,
    dvdt_bc,
    bc_unsteady = true,
    bc_type = (;
        u = (; x = (:dirichlet, :pressure), y = (:symmetric, :symmetric)),
        v = (; x = (:dirichlet, :symmetric), y = (:pressure, :pressure)),
    ),
    T,
)

x = stretched_grid(0.0, 10.0, 200)
y = stretched_grid(-2.0, 2.0, 80)
grid = Grid(x, y; boundary_conditions, T);

plot_grid(grid)

xc, yc = 2.0, 0.0 # Disk center
D = 1.0 # Disk diameter
δ = 0.11 # Disk thickness
Cₜ = 0.01 # Thrust coefficient
inside(x, y) = abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2
bodyforce_u(x, y) = -Cₜ * inside(x, y)
bodyforce_v(x, y) = 0.0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

setup = Setup(; viscosity_model, convection_model, grid, force, boundary_conditions)

pressure_solver = DirectPressureSolver(setup)
# pressure_solver = CGPressureSolver(setup)
# pressure_solver = FourierPressureSolver(setup)

t_start, t_end = tlims = (0.0, 4π)

initial_velocity_u(x, y) = 1.0
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
plotter = RealTimePlotter(;
    nupdate = 1,
    fieldname = :vorticity,
    type = heatmap,
    # type = contour,
    # type = contourf,
)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
# processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44P2(); pressure_solver, Δt = 4π / 200, processors);

plot_tracers(tracer)

plot_pressure(setup, p)

plot_velocity(setup, V, t_end)

plot_vorticity(setup, V, tlims[2])

plot_streamfunction(setup, V, tlims[2])

plot_force(setup, setup.force.F, t_end)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

