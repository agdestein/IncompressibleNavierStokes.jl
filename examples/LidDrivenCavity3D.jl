# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Lid-Driven Cavity - 3D
#
# In this example we consider a box with a moving lid. The velocity is initially at rest. The
# solution should reach at steady state equilibrium after a certain time. The same steady
# state should be obtained when solving a steady state problem.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "LidDrivenCavity3D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 1000.0)

# Boundary conditions: horizontal movement of the top lid
u_bc(x, y, z, t) = y ≈ 1.0 ? 1.0 : 0.0
v_bc(x, y, z, t) = 0.0
w_bc(x, y, z, t) = y ≈ 1.0 ? 0.2 : 0.0
bc_type = (;
    u = (;
        x = (:dirichlet, :dirichlet),
        y = (:dirichlet, :dirichlet),
        z = (:periodic, :periodic),
    ),
    v = (;
        x = (:dirichlet, :dirichlet),
        y = (:dirichlet, :dirichlet),
        z = (:periodic, :periodic),
    ),
    w = (;
        x = (:dirichlet, :dirichlet),
        y = (:dirichlet, :dirichlet),
        z = (:periodic, :periodic),
    ),
)

# A 3D grid is a Cartesian product of three vectors. Here we refine the grid
# near the walls.
x = cosine_grid(0.0, 1.0, 25)
y = cosine_grid(0.0, 1.0, 25)
z = LinRange(-0.2, 0.2, 10)
plot_grid(x, y, z)

# Build setup and assemble operators
setup = Setup(x, y, z; viscosity_model, u_bc, v_bc, w_bc, bc_type);

# Time interval
t_start, t_end = tlims = (0.0, 0.2)

# Initial conditions
initial_velocity_u(x, y, z) = 0.0
initial_velocity_v(x, y, z) = 0.0
initial_velocity_w(x, y, z) = 0.0
initial_pressure(x, y, z) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    t_start;
    initial_pressure,
);

# Solve steady state problem
V, p = solve_steady_state(setup, V₀, p₀; npicard = 5, maxiter = 15);

# Iteration processors
processors = (
    field_plotter(setup; nupdate = 1),
    ## energy_history_plotter(setup; nupdate = 1),
    ## energy_spectrum_plotter(setup; nupdate = 100),
    ## animator(setup, "vorticity.mkv"; nupdate = 4),
    ## vtk_writer(setup; nupdate = 5, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 1),
);

# Solve unsteady problem
V, p, outputs = solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.001, processors)
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
## plot_streamfunction(setup, V, t_end)
nothing
