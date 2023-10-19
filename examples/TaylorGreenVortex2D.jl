# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Taylor-Green vortex - 2D
#
# In this example we consider the Taylor-Green vortex.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "TaylorGreenVortex2D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(2_000)

# A 2D grid is a Cartesian product of two vectors
n = 128
lims = T(0), T(2π)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
plot_grid(x...)

# Build setup and assemble operators
setup = Setup(x...; Re, ArrayType);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
pressure_solver = SpectralPressureSolver(setup)

# Time interval
t_start, t_end = tlims = T(0), T(5)

# Initial conditions
initial_velocity = (
    (x, y) -> -sin(x) * cos(y),
    (x, y) -> cos(x) * sin(y),
)
u₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity,
    t_start;
    pressure_solver,
);

# Solve steady state problem
## u, p = solve_steady_state(setup, u₀, p₀; npicard = 2);

# Iteration processors
processors = (
    ## field_plotter(setup; nupdate = 1),
    energy_history_plotter(setup; nupdate = 1),
    ## energy_spectrum_plotter(setup; nupdate = 1),
    ## animator(setup, "vorticity.mkv"; nupdate = 4),
    ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 1),
);

# Solve unsteady problem
u, p, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    tlims;
    Δt = T(0.01),
    processors,
    pressure_solver,
    inplace = true,
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`

# Export to VTK
save_vtk(setup, u, p, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, u)

# Plot vorticity
plot_vorticity(setup, u)

# Plot streamfunction
## plot_streamfunction(setup, u)
nothing

# Energy history
outputs[1]
