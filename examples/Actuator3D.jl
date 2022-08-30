# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Unsteady actuator case - 3D
#
# In this example, an unsteady inlet velocity profile at encounters a wind
# turbine blade in a wall-less domain. The blade is modeled as a uniform body
# force on a short cylinder.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "Actuator3D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 100.0)

# Boundary conditions: Unsteady BC requires time derivatives
u_bc(x, y, z, t) = x ≈ 0.0 ? cos(π / 6 * sin(π / 6 * t)) : 0.0
v_bc(x, y, z, t) = x ≈ 0.0 ? sin(π / 6 * sin(π / 6 * t)) : 0.0
w_bc(x, y, z, t) = 0.0
dudt_bc(x, y, z, t) =
    x ≈ 0.0 ? -(π / 6)^2 * cos(π / 6 * t) * sin(π / 6 * sin(π / 6 * t)) : 0.0
dvdt_bc(x, y, z, t) =
    x ≈ 0.0 ? (π / 6)^2 * cos(π / 6 * t) * cos(π / 6 * sin(π / 6 * t)) : 0.0
dwdt_bc(x, y, z, t) = 0.0
bc_type = (;
    u = (;
        x = (:dirichlet, :pressure),
        y = (:symmetric, :symmetric),
        z = (:symmetric, :symmetric),
    ),
    v = (;
        x = (:dirichlet, :symmetric),
        y = (:pressure, :pressure),
        z = (:symmetric, :symmetric),
    ),
    w = (;
        x = (:dirichlet, :symmetric),
        y = (:symmetric, :symmetric),
        z = (:pressure, :pressure),
    ),
)

# A 3D grid is a Cartesian product of three vectors
x = stretched_grid(0.0, 6.0, 30)
y = stretched_grid(-2.0, 2.0, 40)
z = stretched_grid(-2.0, 2.0, 40)
plot_grid(x, y, z)

# Actuator body force: A thrust coefficient `Cₜ` distributed over a short cylinder
cx, cy, cz = 2.0, 0.0, 0.0 # Disk center
D = 1.0                    # Disk diameter
δ = 0.11                   # Cylinder height
Cₜ = 5e-4                  # Thrust coefficient
cₜ = Cₜ / (π * (D / 2)^2 * δ)
inside(x, y, z) = abs(x - cx) ≤ δ / 2 && (y - cy)^2 + (z - cz)^2 ≤ (D / 2)^2
bodyforce_u(x, y, z) = -cₜ * inside(x, y, z)
bodyforce_v(x, y, z) = 0.0
bodyforce_w(x, y, z) = 0.0

# Build setup and assemble operators
setup = Setup(
    x,
    y,
    z;
    viscosity_model,
    u_bc,
    v_bc,
    w_bc,
    dudt_bc,
    dvdt_bc,
    dwdt_bc,
    bc_type,
    bodyforce_u,
    bodyforce_v,
    bodyforce_w,
);

# Time interval
t_start, t_end = tlims = (0.0, 3.0)

# Initial conditions (extend inflow)
initial_velocity_u(x, y, z) = 1.0
initial_velocity_v(x, y, z) = 0.0
initial_velocity_w(x, y, z) = 0.0
initial_pressure(x, y, z) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure,
);

# Solve steady state problem
problem = SteadyStateProblem(setup, V₀, p₀);
V, p = solve(problem);

# Iteration processors
logger = Logger(; nupdate = 1)
plotter = RealTimePlotter(; nupdate = 5, fieldname = :velocity)
writer = VTKWriter(; nupdate = 2, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate = 1)
## processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

# Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = solve(problem, RK44P2(); Δt = 0.05, processors, inplace = true);
#hide current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(V, p, t_end, setup, "output/solution")

# Plot tracers
plot_tracers(tracer)

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V, t_end)

# Plot vorticity
plot_vorticity(setup, V, t_end)

# Plot streamfunction
## plot_streamfunction(setup, V, t_end)

# Plot force
plot_force(setup, t_end)
