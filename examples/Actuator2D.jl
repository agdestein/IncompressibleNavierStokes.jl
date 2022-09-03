# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Unsteady actuator case - 2D
#
# In this example, an unsteady inlet velocity profile at encounters a wind
# turbine blade in a wall-less domain. The blade is modeled as a uniform body
# force on a thin rectangle.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "Actuator2D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 100.0)

# Boundary conditions: Unsteady BC requires time derivatives
u_bc(x, y, t) = x ≈ 0.0 ? cos(π / 6 * sin(π / 6 * t)) : 0.0
v_bc(x, y, t) = x ≈ 0.0 ? sin(π / 6 * sin(π / 6 * t)) : 0.0
dudt_bc(x, y, t) = x ≈ 0.0 ? -(π / 6)^2 * cos(π / 6 * t) * sin(π / 6 * sin(π / 6 * t)) : 0.0
dvdt_bc(x, y, t) = x ≈ 0.0 ? (π / 6)^2 * cos(π / 6 * t) * cos(π / 6 * sin(π / 6 * t)) : 0.0
bc_type = (;
    u = (; x = (:dirichlet, :pressure), y = (:symmetric, :symmetric)),
    v = (; x = (:dirichlet, :symmetric), y = (:pressure, :pressure)),
)

# A 2D grid is a Cartesian product of two vectors
n = 40
x = LinRange(0.0, 10.0, 5n)
y = LinRange(-2.0, 2.0, 2n)
plot_grid(x, y)

# Actuator body force: A thrust coefficient `Cₜ` distributed over a thin rectangle
xc, yc = 2.0, 0.0 # Disk center
D = 1.0           # Disk diameter
δ = 0.11          # Disk thickness
Cₜ = 5e-4         # Thrust coefficient
cₜ = Cₜ / (D * δ)
inside(x, y) = abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2
bodyforce_u(x, y) = -cₜ * inside(x, y)
bodyforce_v(x, y) = 0.0

# Build setup and assemble operators
setup = Setup(
    x,
    y;
    viscosity_model,
    u_bc,
    v_bc,
    dudt_bc,
    dvdt_bc,
    bc_type,
    bodyforce_u,
    bodyforce_v,
);

# Time interval
t_start, t_end = tlims = (0.0, 12.0)

# Initial conditions (extend inflow)
initial_velocity_u(x, y) = 1.0
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
);

# Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = solve(problem);

# Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 1, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 2, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
## processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

# Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = solve(problem, RK44P2(); Δt = 0.05, processors, inplace = true);
#md current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`.

# We create a box to visualize the actuator.
box = (
    [xc - δ / 2, xc - δ / 2, xc + δ / 2, xc + δ / 2, xc - δ / 2],
    [yc + D / 2, yc - D / 2, yc - D / 2, yc + D / 2, yc + D / 2],
)

# Export to VTK
save_vtk(V, p, t_end, setup, "output/solution")

# Plot tracers
plot_tracers(tracer)

# Plot pressure
fig = plot_pressure(setup, p)
lines!(box...; color = :red)
fig

# Plot velocity
fig = plot_velocity(setup, V, t_end)
lines!(box...; color = :red)
fig

# Plot vorticity
fig = plot_vorticity(setup, V, t_end)
lines!(box...; color = :red)
fig

# Plot streamfunction
fig = plot_streamfunction(setup, V, t_end)
lines!(box...; color = :red)
fig

# Plot force
fig = plot_force(setup, t_end)
lines!(box...; color = :red)
fig
