using IncompressibleNavierStokes

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

name = "TaylorGreenVortex3D"

T = Float64

viscosity_model = LaminarModel{T}(; Re = 2000)
# viscosity_model = KEpsilonModel{T}(; Re = 1000)
# viscosity_model = MixingLengthModel{T}(; Re = 1000)
# viscosity_model = SmagorinskyModel{T}(; Re = 1000)
# viscosity_model = QRModel{T}(; Re = 1000)

convection_model = NoRegConvectionModel()
# convection_model = C2ConvectionModel()
# convection_model = C4ConvectionModel()
# convection_model = LerayConvectionModel()

u_bc(x, y, z, t) = 0.0
v_bc(x, y, z, t) = 0.0
w_bc(x, y, z, t) = 0.0
boundary_conditions = BoundaryConditions(
    u_bc,
    v_bc,
    w_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (;
            x = (:periodic, :periodic),
            y = (:periodic, :periodic),
            z = (:periodic, :periodic),
        ),
        v = (;
            x = (:periodic, :periodic),
            y = (:periodic, :periodic),
            z = (:periodic, :periodic),
        ),
        w = (;
            x = (:periodic, :periodic),
            y = (:periodic, :periodic),
            z = (:periodic, :periodic),
        ),
    ),
    T,
)

x = stretched_grid(0, 2π, 20)
y = stretched_grid(0, 2π, 20)
z = stretched_grid(0, 2π, 20)
grid = Grid(x, y, z; boundary_conditions, T);

plot_grid(grid)

bodyforce_u(x, y, z) = 0.0
bodyforce_v(x, y, z) = 0.0
bodyforce_w(x, y, z) = 0.0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, bodyforce_w, grid)

setup = Setup(; viscosity_model, convection_model, grid, force, boundary_conditions)

# pressure_solver = DirectPressureSolver(setup)
# pressure_solver = CGPressureSolver(setup)
pressure_solver = FourierPressureSolver(setup)

t_start, t_end = tlims = (0.0, 10.0)

initial_velocity_u(x, y, z) = sin(x)cos(y)cos(z)
initial_velocity_v(x, y, z) = -cos(x)sin(y)cos(z)
initial_velocity_w(x, y, z) = 0.0
initial_pressure(x, y, z) = 1 / 4 * (cos(2x) + cos(2y) + cos(2z))
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure,
    pressure_solver,
);

problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem; npicard = 6)

logger = Logger()
plotter = RealTimePlotter(; nupdate = 10, fieldname = :vorticity, type = contour)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
# processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = @time solve(problem, RK44(); Δt = 0.01, processors, pressure_solver)

plot_tracers(tracer)

plot_pressure(setup, p; alpha = 0.05)

plot_velocity(setup, V, t_end; alpha = 0.05)

plot_vorticity(setup, V, tlims[2]; alpha = 0.05)

# plot_streamfunction(setup, V, tlims[2])

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

