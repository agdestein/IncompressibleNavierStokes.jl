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

# A 2D grid is a Cartesian product of two vectors
n = 40
x = LinRange(0.0, 10.0, 5n + 1)
y = LinRange(-2.0, 2.0, 2n + 1)
plot_grid(x, y)

# Boundary conditions
boundary_conditions = (
    ## x left, x right
    (
        ## Unsteady BC requires time derivatives
        DirichletBC(
            (dim, x, y, t) -> sin(π / 6 * sin(π / 6 * t) + π / 2 * (dim() == 1)),
            (dim, x, y, t) ->
                (π / 6)^2 *
                cos(π / 6 * t) *
                cos(π / 6 * sin(π / 6 * t) + π / 2 * (dim() == 1)),
        ),
        PressureBC(),
    ),

    ## y rear, y front
    (PressureBC(), PressureBC()),
)

# Actuator body force: A thrust coefficient `Cₜ` distributed over a thin rectangle
xc, yc = 2.0, 0.0 # Disk center
D = 1.0           # Disk diameter
δ = 0.11          # Disk thickness
Cₜ = 0.2          # Thrust coefficient
cₜ = Cₜ / (D * δ)
inside(x, y) = abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2
bodyforce(dim, x, y, t) = dim() == 1 ? -cₜ * inside(x, y) : 0.0

# Build setup and assemble operators
setup = Setup(x, y; Re = 100.0, boundary_conditions, bodyforce);

# Initial conditions (extend inflow)
u₀, p₀ = create_initial_conditions(setup, (dim, x, y) -> dim() == 1 ? 1.0 : 0.0);
u, p = u₀, p₀

# Solve unsteady problem
u, p, outputs = solve_unsteady(
    setup,
    ## u₀, p₀,
    u,
    p,
    (0.0, 12.0);
    method = RK44P2(),
    Δt = 0.05,
    processors = (
        field_plotter(setup; nupdate = 1),
        ## energy_history_plotter(setup; nupdate = 10),
        ## energy_spectrum_plotter(setup; nupdate = 10),
        ## animator(setup, "vorticity.mkv"; nupdate = 4),
        ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
        ## field_saver(setup; nupdate = 10),
        step_logger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`.

# We create a box to visualize the actuator.
box = (
    [xc - δ / 2, xc - δ / 2, xc + δ / 2, xc + δ / 2, xc - δ / 2],
    [yc + D / 2, yc - D / 2, yc - D / 2, yc + D / 2, yc + D / 2],
)

# Export to VTK
save_vtk(setup, u, p, "output/solution")

# Field plot
fig = outputs[1]
lines!(box...; color = :red)
fig

# Plot pressure
fig = plot_pressure(setup, p)
lines!(box...; color = :red)
fig

# Plot velocity
fig = plot_velocity(setup, u)
lines!(box...; color = :red)
fig

# Plot vorticity
fig = plot_vorticity(setup, u)
lines!(box...; color = :red)
fig

# Plot streamfunction
fig = plot_streamfunction(setup, u)
lines!(box...; color = :red)
fig

# Plot force
fig = plot_force(setup)
lines!(box...; color = :red)
fig
