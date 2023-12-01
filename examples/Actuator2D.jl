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

# Output directory
output = "output/Actuator2D"

# A 2D grid is a Cartesian product of two vectors
n = 40
x = LinRange(0.0, 10.0, 5n + 1)
y = LinRange(-2.0, 2.0, 2n + 1)
plotgrid(x, y)

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
state, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    (0.0, 12.0);
    method = RK44P2(),
    Δt = 0.05,
    processors = (
        rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 1),
        ## ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 1),
        ## espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 1),
        ## anim = animator(; setup, path = "$output/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = "$output", filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`.

# Export to VTK
save_vtk(setup, state.u, state.p, "$output/solution")

# We create a box to visualize the actuator.
box = (
    [xc - δ / 2, xc - δ / 2, xc + δ / 2, xc + δ / 2, xc - δ / 2],
    [yc + D / 2, yc - D / 2, yc - D / 2, yc + D / 2, yc + D / 2],
)

# Plot pressure
fig = fieldplot(state; setup, fieldname = :pressure)
lines!(box...; color = :red)
fig

# Plot velocity
fig = fieldplot(state; setup, fieldname = :velocity)
lines!(box...; color = :red)
fig

# Plot vorticity
fig = fieldplot(state; setup, fieldname = :vorticity)
lines!(box...; color = :red)
fig
