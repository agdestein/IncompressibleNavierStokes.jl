# # Lid-Driven Cavity case (LDC)
#
# This test case considers a box with a moving lid, where the velocity is initially at rest.

# LSP indexing solution                                                          #src
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983 #src
if isdefined(@__MODULE__, :LanguageServer)                                       #src
    include("../src/IncompressibleNavierStokes.jl")                              #src
    using .IncompressibleNavierStokes                                            #src
end                                                                              #src

# We start by loading IncompressibleNavierStokes and a Makie plotting backend.

using IncompressibleNavierStokes

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

# ## Build problem

# We may choose the floating point type for the simulations. Replacing `Float64` with
# `Float32` will not necessarilily speed up the simulations, but requires half as much
# memory.

T = Float64

# Available viscosity models are:
#
# - [`LaminarModel`](@ref),
# - [`KEpsilonModel`](@ref),
# - [`MixingLengthModel`](@ref),
# - [`SmagorinskyModel`](@ref), and
# - [`QRModel`](@ref).
#
# They all take a Reynolds number as a parameter. Here we choose a moderate Reynolds number.

viscosity_model = LaminarModel{T}(; Re = 1000)
# viscosity_model = KEpsilonModel{T}(; Re = 3000)
# viscosity_model = MixingLengthModel{T}(; Re = 3000)
# viscosity_model = SmagorinskyModel{T}(; Re = 3000)
# viscosity_model = QRModel{T}(; Re = 3000)

# Available convection models are:
#
# - [`NoRegConvectionModel`](@ref),
# - [`C2ConvectionModel`](@ref),
# - [`C4ConvectionModel`](@ref), and
# - [`LerayConvectionModel`](@ref).
#
# We here take the simplest model.

convection_model = NoRegConvectionModel{T}()

# We create a two-dimensional domain with a box of size `[1, 1]`.

Nx = 100                          # Number of x-volumes
Ny = 100                          # Number of y-volumes
grid = create_grid(
    T,
    Nx,
    Ny;
    xlims = (0, 1),               # Horizontal limits (left, right)
    ylims = (0, 1),               # Vertical limits (bottom, top)
    stretch = (1, 1),             # Stretch factor (sx, sy)
    order4 = false,               # Use 4th order in space
)

# Solver settings are used by certain implicit solvers.

solver_settings = SolverSettings{T}(;
    pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
    p_add_solve = true,                             # Additional pressure solve for second order pressure
    abstol = 1e-10,                                 # Absolute accuracy
    reltol = 1e-14,                                 # Relative accuracy
    maxiter = 10,                                   # Maximum number of iterations
    newton_type = :approximate,
)

# Dirichlet boundary conditions are specified as plain Julia functions. They are marked by
# the `:dirichlet` symbol. Other possible BC types are `:periodic`, `:symmetric`, and `:pressure`.

u_bc(x, y, t, setup) = y ≈ setup.grid.ylims[2] ? 1.0 : 0.0
v_bc(x, y, t, setup) = zero(x)
bc = create_boundary_conditions(
    T,
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        k = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        e = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ν = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
    ),
)

# The body forces are specified as plain Julia functions.

bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

# We may now assemble our setup.

setup = Setup{T,2}(; viscosity_model, convection_model, grid, force, solver_settings, bc)

# The discrete operators are built with the [`build_operators!`](@ref) function.

build_operators!(setup);

# We will solve for a time interval of ten seconds.

t_start, t_end = tlims = (0.0, 10.0)

# The initial conditions are defined as plain Julia functions.

initial_velocity_u(x, y) = 0
initial_velocity_v(x, y) = 0
initial_pressure(x, y) = 0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
);


# ## Solve problems
#
# There are many different problems. They can all be solved by calling the [`solve`](@ref)
# function.
#
# A [`SteadyStateProblem`](@ref) is for computing a state where the right hand side of the
# momentum equation is zero.

problem = SteadyStateProblem(setup, V₀, p₀);
V, p = @time solve(problem);

# For this test case, the same steady state may be obtained by solving an
# [`UnsteadyProblem`](@ref) for a sufficiently long time.
problem = UnsteadyProblem(setup, V₀, p₀, tlims);

# We may also define a list of iteration processors. They are processed after every
# `nupdate` iteration.

logger = Logger(; nupdate = 10)
plotter = RealTimePlotter(; nupdate = 5, fieldname = :vorticity)
writer = VTKWriter(; nupdate = 10, dir = "output/LidDrivenCavity2D")
tracer = QuantityTracer(; nupdate = 1)
processors = [logger, plotter, tracer]

#  A ODE method is needed. Here we will opt for a standard fourth order Runge-Kutta method
#  with a fixed time step.

V, p = @time solve(problem, RK44(); Δt = 0.01, processors);


# ## Postprocess
#
# The `tracer` object contains a history of some quantities related to the momentum and
# energy.

plot_tracers(tracer)

# We may also plot the final pressure field,

plot_pressure(setup, p)

# velocity field,

plot_velocity(setup, V, t_end)

# vorticity field,

plot_vorticity(setup, V, tlims[2])

# or streamfunction.

plot_streamfunction(setup, V, tlims[2])
