# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Backward Facing Step - 2D
#
# In this example we consider a channel with walls at the top and bottom, and a
# step at the left with a parabolic inflow. Initially the velocity is an
# extension of the inflow, but as time passes the velocity finds a new steady
# state.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "BackwardFacingStep2D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 3000.0)

# Boundary conditions: steady inflow on the top half
u_bc(x, y, t) = x ≈ 0 && y ≥ 0 ? 24y * (1 / 2 - y) : 0.0
v_bc(x, y, t) = 0.0
bc_type = (;
    u = (; x = (:dirichlet, :pressure), y = (:dirichlet, :dirichlet)),
    v = (; x = (:dirichlet, :symmetric), y = (:dirichlet, :dirichlet)),
)

# A 2D grid is a Cartesian product of two vectors. Here we refine the grid near
# the walls.
x = LinRange(0.0, 10.0, 300)
y = cosine_grid(-0.5, 0.5, 50)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model, u_bc, v_bc, bc_type);

# Time interval
t_start, t_end = tlims = (0.0, 7.0)

# Initial conditions (extend inflow)
initial_velocity_u(x, y) = y ≥ 0.0 ? 24y * (1 / 2 - y) : 0.0
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity_u,
    initial_velocity_v,
    t_start;
    initial_pressure,
);

# Solve steady state problem
V, p = solve_steady_state(setup, V₀, p₀);

# Iteration processors
processors = (
    field_plotter(setup; nupdate = 5),
    ## energy_history_plotter(setup; nupdate = 10),
    ## energy_spectrum_plotter(setup; nupdate = 10),
    ## animator(setup, "vorticity.mkv"; nupdate = 4),
    ## vtk_writer(setup; nupdate = 20, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 1),
);

# Solve unsteady problem
V, p, outputs = solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.002, processors, inplace = true)
#md current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(setup, V, p, t_end, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V, t_end)

# Plot vorticity
plot_vorticity(setup, V, t_end)

# Plot streamfunction
plot_streamfunction(setup, V, t_end)
