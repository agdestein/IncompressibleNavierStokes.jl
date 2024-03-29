# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Planar mixing - 2D
#
# Planar mixing example, as presented in [List2022](@cite).

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "PlanarMixing2D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 500.0)

# Boundary conditions: Unsteady BC requires time derivatives
ΔU = 1.0
Ū = 1.0
ϵ = (0.082Ū, 0.012Ū)
n = (0.4π, 0.3π)
ω = (0.22, 0.11)
u_bc(x, y, t) =
    x ≈ 0.0 ?
    1.0 + ΔU / 2 * tanh(2y) + sum(@. ϵ * (1 - tanh(y / 2)^2) * cos(n * y) * sin(ω * t)) :
    0.0
v_bc(x, y, t) = 0.0
dudt_bc(x, y, t) =
    x ≈ 0.0 ? sum(@. ϵ * (1 - tanh(y / 2)^2) * cos(n * y) * ω * cos(ω * t)) : 0.0
dvdt_bc(x, y, t) = 0.0
bc_type = (;
    u = (; x = (:dirichlet, :pressure), y = (:symmetric, :symmetric)),
    v = (; x = (:dirichlet, :symmetric), y = (:pressure, :pressure)),
)

# A 2D grid is a Cartesian product of two vectors
n = 64
## n = 256
x = LinRange(0.0, 256.0, 4n)
y = LinRange(-32.0, 32.0, n)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model, u_bc, v_bc, dudt_bc, dvdt_bc, bc_type);

# Time interval
t_start, t_end = tlims = (0.0, 100.0)

# Initial conditions
initial_velocity_u(x, y) = u_bc(0.0, y, 0.0)
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity_u,
    initial_velocity_v,
    t_start;
    initial_pressure,
);

# Iteration processors
processors = (
    field_plotter(setup; nupdate = 1),
    ## energy_history_plotter(setup; nupdate = 1),
    ## energy_spectrum_plotter(setup; nupdate = 100),
    ## animator(setup, "vorticity.mkv"; nupdate = 4),
    ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 1),
);

# Solve unsteady problem
V, p, outputs = solve_unsteady(
    setup,
    V₀,
    p₀,
    tlims;
    method = RK44P2(),
    Δt = 0.1,
    processors,
    inplace = true,
);
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
