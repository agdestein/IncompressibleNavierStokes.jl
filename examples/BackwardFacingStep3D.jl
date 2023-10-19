# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Backward Facing Step - 3D
#
# In this example we consider a channel with periodic side boundaries, walls at
# the top and bottom, and a step at the left with a parabolic inflow. Initially
# the velocity is an extension of the inflow, but as time passes the velocity
# finds a new steady state.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "BackwardFacingStep3D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(3000)

# A 3D grid is a Cartesian product of three vectors
x = LinRange(T(0), T(10), 129)
y = LinRange(-T(0.5), T(0.5), 17)
z = LinRange(-T(0.25), T(0.25), 9)
plot_grid(x, y, z)

# Boundary conditions: steady inflow on the top half
U(x, y, z, t) = y ≥ 0 ? 24y * (1 - y) / 2 : zero(x)
V(x, y, z, t) = zero(x)
W(x, y, z, t) = zero(x)
dUdt(x, y, z, t) = zero(x)
dVdt(x, y, z, t) = zero(x)
dWdt(x, y, z, t) = zero(x)
boundary_conditions = (
    ## x left, x right
    (DirichletBC((U, V, W), (dUdt, dVdt, dWdt)), PressureBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC()),

    ## z bottom, z top
    (PeriodicBC(), PeriodicBC()),
)

# Build setup and assemble operators
setup = Setup(x, y, z; Re, boundary_conditions, ArrayType);

# Time interval
t_start, t_end = tlims = T(0), T(7)

# Initial conditions (extend inflow)
initial_velocity = (
    (x, y, z) -> U(x, y, z, zero(x)),
    (x, y, z) -> zero(x),
    (x, y, z) -> zero(x),
)
u₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity,
    t_start;
);

# Solve steady state problem
## u, p = solve_steady_state(setup, u₀, p₀);

# Iteration processors
processors = (
    field_plotter(setup; nupdate = 50),
    ## energy_history_plotter(setup; nupdate = 10),
    ## energy_spectrum_plotter(setup; nupdate = 10),
    ## animator(setup, "vorticity.mkv"; nupdate = 4),
    ## vtk_writer(setup; nupdate = 20, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 10),
);

# Solve unsteady problem
u, p, outputs = solve_unsteady(setup, u₀, p₀, tlims; Δt = 0.01, processors, inplace = true)
#md current_figure()

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
