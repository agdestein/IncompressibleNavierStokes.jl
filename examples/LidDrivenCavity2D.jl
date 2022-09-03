# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Lid-Driven Cavity - 2D
#
# In this example we consider a box with a moving lid. The velocity is
# initially at rest. The solution should reach at steady state equilibrium
# after a certain time. The same steady state should be obtained when solving a
# [`SteadyStateProblem`](@ref).

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "LidDrivenCavity2D"

# Available viscosity models are:
#
# - [`LaminarModel`](@ref),
# - [`KEpsilonModel`](@ref),
# - [`MixingLengthModel`](@ref),
# - [`SmagorinskyModel`](@ref), and
# - [`QRModel`](@ref).
#
# They all take a Reynolds number as a parameter. Here we choose a moderate
# Reynolds number.
viscosity_model = LaminarModel(; Re = 1000.0)

# Dirichlet boundary conditions are specified as plain Julia functions. They
# are marked by the `:dirichlet` symbol. Other possible BC types are
# `:periodic`, `:symmetric`, and `:pressure`.
u_bc(x, y, t) = y ≈ 1.0 ? 1.0 : 0.0
v_bc(x, y, t) = 0.0
bc_type = (;
    u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
)

# We create a two-dimensional domain with a box of size `[1, 1]`. The grid is
# created as a Cartesian product between two vectors. We add a refinement near
# the walls.
x = cosine_grid(0.0, 1.0, 40)
y = cosine_grid(0.0, 1.0, 40)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model, u_bc, v_bc, bc_type);

# We will solve for a time interval of ten seconds.
t_start, t_end = tlims = (0.0, 10.0)

# The initial conditions are defined as plain Julia functions.
initial_velocity_u(x, y) = 0.0
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
)

# ## Solve problems
#
# Problems can be solved solved by calling the [`solve`](@ref) function.

# A [`SteadyStateProblem`](@ref) is for computing a state where the right hand side of the
# momentum equation is zero.
problem = SteadyStateProblem(setup, V₀, p₀)
V, p = solve(problem)

# For this test case, the same steady state may be obtained by solving an
# [`UnsteadyProblem`](@ref) for a sufficiently long time.
problem = UnsteadyProblem(setup, V₀, p₀, tlims)

# We may also define a list of iteration processors. They are processed after every
# `nupdate` iteration.
logger = Logger(; nupdate = 1000)
plotter = RealTimePlotter(; nupdate = 50, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 20, dir = "output/LidDrivenCavity2D")
tracer = QuantityTracer(; nupdate = 10)
## processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

# A ODE method is needed. Here we will opt for a standard fourth order Runge-Kutta method
# with a fixed time step.
V, p = solve(problem, RK44(); Δt = 0.001, processors)
#md current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export fields to VTK. The file `output/solution.vtr` may be opened for visulization
# in [ParaView](https://www.paraview.org/).
save_vtk(V, p, t_end, setup, "output/solution")

# The `tracer` object contains a history of some quantities related to the
# momentum and energy.
plot_tracers(tracer)

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V, t_end)

# Plot vorticity (with custom levels)
levels = [-7, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 7]
plot_vorticity(setup, V, t_end; levels)

# Plot streamfunction
plot_streamfunction(setup, V, t_end)
