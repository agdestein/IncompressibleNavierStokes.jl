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

# The code allows for using different floating point number types, including single
# precision (`Float32`) and double precision (`Float64`). On the CPU, the speed
# is not really different, but double precision uses twice as much memory as
# single precision. When running on the GPU, single precision is preferred.
# Half precision (`Float16`) is also an option, but then the values should be
# scaled judiciously to avoid vanishing digits when applying differential
# operators of the form "right minus left divided by small distance".
# T = Float64
T = Float32

# Available viscosity models are:
#
# - [`LaminarModel`](@ref),
# - [`MixingLengthModel`](@ref),
# - [`SmagorinskyModel`](@ref), and
# - [`QRModel`](@ref).
#
# They all take a Reynolds number as a parameter. Here we choose a moderate
# Reynolds number. Note how we pass the floating point type.
viscosity_model = LaminarModel(; Re = T(1_000))

# Dirichlet boundary conditions are specified as plain Julia functions. They
# are marked by the `:dirichlet` symbol. Other possible BC types are
# `:periodic`, `:symmetric`, and `:pressure`.
u_bc(x, y, t) = y ≈ 1 ? T(1) : T(0)
v_bc(x, y, t) = T(0)
bc_type = (;
    u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
)

# We create a two-dimensional domain with a box of size `[1, 1]`. The grid is
# created as a Cartesian product between two vectors. We add a refinement near
# the walls.
n = 256
lims = (T(0), T(1))
# x = cosine_grid(lims..., n)
# y = cosine_grid(lims..., n)
x = LinRange(lims..., n)
y = LinRange(lims..., n)
plot_grid(x, y)

# We can now build the setup and assemble operators. A 3D setup is built if we
# also provide a vector of z-coordinates.
setup = Setup(x, y; viscosity_model, u_bc, v_bc, bc_type);

# The pressure solver is used to solve the pressure poisson equation.
pressure_solver = CGPressureSolver(setup)

# We will solve for a time interval of ten seconds.
t_start, t_end = tlims = (T(0), T(10))

# The initial conditions are defined as plain Julia functions.
initial_velocity_u(x, y) = 0
initial_velocity_v(x, y) = 0
initial_pressure(x, y) = 0
V₀, p₀ = create_initial_conditions(
    setup,
    initial_velocity_u,
    initial_velocity_v,
    t_start;
    initial_pressure,
    pressure_solver,
)

# ## Solve problems
#
# Problems can be solved.

# The [`solve_steady_state`](@ref) function is for computing a state where the right hand side of the
# momentum equation is zero.
V, p = solve_steady_state(setup, V₀, p₀)

# For this test case, the same steady state may be obtained by solving an
# unsteady problem for a sufficiently long time.

# Iteration processors
processors = (
    # field_plotter(setup; nupdate = 1),
    step_logger(; nupdate = 1),
    # vtk_writer(setup; nupdate = 20, dir = "output/$name", filename = "solution"),
);

# Iteration processors
processors = (
    # field_plotter(setup; nupdate = 1),
    # energy_history_plotter(setup; nupdate = 1),
    # energy_spectrum_plotter(setup; nupdate = 100),
    # animator(setup, "vorticity.mkv"; nupdate = 4),
    # vtk_writer(setup; nupdate = 20, dir = "output/$name", filename = "solution"),
    step_logger(; nupdate = 1),
);

# By default, a standard fourth order Runge-Kutta method is used. If we don't
# provide the time step explicitly, an adaptive time step is used.
V, p, outputs = solve_unsteady(setup, V₀, p₀, tlims; Δt = T(0.00001), processors, pressure_solver);

# GPU
using CUDA
cusetup = cu(setup)
processors = (
    # field_plotter(cusetup; nupdate = 1),
    step_logger(; nupdate = 1),
);
V, p, outputs = solve_unsteady(
    cusetup, cu(V₀), cu(p₀), tlims;
    Δt = T(0.00001),
    processors,
    pressure_solver = CGPressureSolver(cusetup),
    inplace = false,
    bc_vectors = cu(get_bc_vectors(setup, t_start)),
);

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export fields to VTK. The file `output/solution.vtr` may be opened for visulization
# in [ParaView](https://www.paraview.org/).
save_vtk(setup, V, p, t_end, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V, t_end)

# Plot vorticity (with custom levels)
levels = [-7, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 7]
plot_vorticity(setup, V, t_end; levels)

# Plot streamfunction
plot_streamfunction(setup, V, t_end)

# In addition, the tuple `outputs` contains quantities from our processors.
# The logger returns nothing.
outputs[1]

# The [`vtk_writer`](@ref) returns the file name of the ParaView collection
# file.
outputs[2]

# The [`field_plotter`](@ref) returns the field plot figure.
outputs[3]
