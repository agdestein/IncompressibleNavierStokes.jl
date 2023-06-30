# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Decaying Homogeneous Isotropic Turbulence - 3D
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

using FFTW
#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "DecayingTurbulence3D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 1e4)

# A 3D grid is a Cartesian product of three vectors
n = 50
x = LinRange(0.0, 1.0, n + 1)
y = LinRange(0.0, 1.0, n + 1)
z = LinRange(0.0, 1.0, n + 1)
plot_grid(x, y, z)

# Build setup and assemble operators
setup = Setup(x, y, z; viscosity_model);

# Since the grid is uniform and identical for x, y, and z, we may use a
# specialized Fourier pressure solver
pressure_solver = FourierPressureSolver(setup)

# Initial conditions
K = n ÷ 2
V₀, p₀ = random_field(
    setup, K;
    A = 1e6,
    σ = 30,
    ## σ = 10,
    s = 5,
    pressure_solver,
)

# Time interval
t_start, t_end = tlims = (0.0, 0.1)

# Iteration processors
logger = Logger()
observer = StateObserver(1, V₀, p₀, t_start)
writer = VTKWriter(; nupdate = 100, dir = "output/$name", filename = "solution")
tracer = QuantityTracer()
## processors = [logger, observer, tracer, writer]
processors = [logger, observer, tracer]

# Real time plot
rtp = real_time_plot(observer, setup)

# Plot energy history
ehist = energy_history_plot(observer, setup)

# Plot energy spectrum
espec = energy_spectrum_plot(observer, setup, K)

# Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = solve(problem, RK44(); Δt = 0.001, processors, pressure_solver, inplace = true);

# Real time plot
rtp

# Energy history
ehist

# Energy spectrum
espec

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
