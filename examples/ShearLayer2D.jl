# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Shear layer - 2D
#
# Shear layer example.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "ShearLayer2D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(Inf)

# A 2D grid is a Cartesian product of two vectors
n = 128
lims = T(0), T(2π)
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; Re, ArrayType);

pressure_solver = SpectralPressureSolver(setup)

# Time interval
t_start, t_end = tlims = T(0), T(8)

# Initial conditions: We add 1 to u in order to make global momentum
# conservation less trivial
d = T(π / 15)
e = T(0.05)
initial_velocity = (
    (x, y) -> y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d),
    (x, y) -> e * sin(x),
)
## initial_velocity = (
##     (x, y) -> T(1) + (y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d)),
##     (x, y) -> e * sin(x),
## )
u₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity,
    t_start;
    pressure_solver,
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
u, p, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    tlims;
    method = RK44(),
    Δt = T(0.01),
    processors,
    inplace = true,
    pressure_solver,
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`

outputs[1]

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
