# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Decaying Homogeneous Isotropic Turbulence - 2D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

# use Makie.inline!(false) to make plots interactive

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "DecayingTurbulence2D_snapshot_production"

# Floating point precision
T = Float32

# To use CPU: Do not move any arrays
device = identity

# To use GPU, use `cu` to move arrays to the GPU.
# Note: `cu` converts to Float32
## using CUDA
## device = cu

# Viscosity model
viscosity_model = LaminarModel(; Re = T(10_000))

# A 2D grid is a Cartesian product of two vectors
n = 256 # 1024
lims = (T(0), T(1))
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
# plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
pressure_solver = SpectralPressureSolver(setup);

K = 256; 

snapshots_V₀ = zeros(setup.grid.NV,K);
snapshots_V  = zeros(setup.grid.NV,K);

for k = 1:K
    k
    # Initial conditions
    V₀, p₀ = random_field(setup; A = T(1_000_000), σ = T(30), s = 5, pressure_solver);

    snapshots_V₀[:,k] = V₀;

    # Iteration processors
    processors = (
        field_plotter(device(setup); nupdate = 1,displayfig = false),
        energy_history_plotter(device(setup); nupdate = 20, displayfig = false),
        energy_spectrum_plotter(device(setup); nupdate = 1, displayfig = false),
        ## animator(device(setup), "vorticity.mp4"; nupdate = 16),
        ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
        ## field_saver(setup; nupdate = 10),
        step_logger(; nupdate = 100),
    );

    # Time interval
    t_start, t_end = tlims = (T(0), T(.5))

    # Solve unsteady problem
    V, p, outputs = solve_unsteady(
        setup,
        V₀,
        p₀,
        tlims;
        Δt = T(0.001),
        processors,
        pressure_solver,
        inplace = true,
        device,
    );

    snapshots_V[:,k] = V;
end

JLD2.save_object("output/snapshots_V.jld2",snapshots_V)
JLD2.save_object("output/snapshots_V_0.jld2",snapshots_V₀)



# Field plot
outputs[1]

# Energy history plot
outputs[2]

# Energy spectrum plot
outputs[3]

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
# save_vtk(setup, V, p, t_end, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V, t_end)

# Plot vorticity
plot_vorticity(setup, V, t_end)
