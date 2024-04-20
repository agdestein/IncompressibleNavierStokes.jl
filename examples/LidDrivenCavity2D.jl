# # Tutorial: Lid-Driven Cavity - 2D
#
# In this example we consider a box with a moving lid. The velocity is
# initially at rest. The solution should reach at steady state equilibrium
# after a certain time. The same steady state should be obtained when solving a
# steady state problem.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
#
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
output = "output/LidDrivenCavity2D"

# The code allows for using different floating point number types, including single
# precision (`Float32`) and double precision (`Float64`). On the CPU, the speed
# is not really different, but double precision uses twice as much memory as
# single precision. When running on the GPU, single precision is preferred.
# Half precision (`Float16`) is also an option, but then the values should be
# scaled judiciously to avoid vanishing digits when applying differential
# operators of the form "right minus left divided by small distance".

# Note how floating point type hygiene is enforced in the following using `T`
# to avoid mixing different precisions.

T = Float64
## T = Float32
## T = Float16

# We can also choose to do the computations on a different device. By default,
# the computations are performed on the host (CPU). An optional `ArrayType`
# allows for moving arrays to a different device such as a GPU.
#
# Note: For GPUs, single precision is preferred.

ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Here we choose a moderate Reynolds number. Note how we pass the floating point type.
Re = T(1_000)

# Non-zero Dirichlet boundary conditions are specified as plain Julia functions.
# Note that time derivatives are required.
boundary_conditions = (
    ## x left, x right
    (DirichletBC(), DirichletBC()),

    ## y bottom, y top
    (
        DirichletBC(),
        DirichletBC(
            (dim, x, y, t) -> dim() == 1 ? one(x) : zero(x),
            (dim, x, y, t) -> zero(x),
        ),
    ),
)

# We create a two-dimensional domain with a box of size `[1, 1]`. The grid is
# created as a Cartesian product between two vectors. We add a refinement near
# the walls.
n = 32
lims = T(0), T(1)
x = cosine_grid(lims..., n)
y = cosine_grid(lims..., n)
plotgrid(x, y)

# We can now build the setup and assemble operators.
# A 3D setup is built if we also provide a vector of z-coordinates.
setup = Setup(x, y; boundary_conditions, Re, ArrayType);

# The pressure solver is used to solve the pressure Poisson equation.
# Available solvers are
#
# - [`DirectPressureSolver`](@ref) (only for CPU with `Float64`)
# - [`CGPressureSolver`](@ref)
# - [`SpectralPressureSolver`](@ref) (only for periodic boundary conditions and
#   uniform grids)

psolver = DirectPressureSolver(setup);

# The initial conditions are provided in function. The value `dim()` determines
# the velocity component.
u₀ = create_initial_conditions(setup, (dim, x, y) -> zero(x); psolver);

# ## Solve problems
#
# Problems can be solved.

# The [`solve_steady_state`](@ref) function is for computing a state where the right hand side of the
# momentum equation is zero.
## u, p = solve_steady_state(setup, u₀, p₀)
nothing

# For this test case, the same steady state may be obtained by solving an
# unsteady problem for a sufficiently long time.

# Iteration processors are called after every `nupdate` time steps. This can be
# useful for logging, plotting, or saving results. Their respective outputs are
# later returned by `solve_unsteady`.

processors = (
    ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 50),
    ## ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 10),
    ## espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
    ## anim = animator(; setup, path = "$output/solution.mkv", nupdate = 20),
    ## vtk = vtk_writer(; setup, nupdate = 100, dir = output, filename = "solution"),
    ## field = fieldsaver(; setup, nupdate = 10),
    log = timelogger(; nupdate = 1000),
);

# By default, a standard fourth order Runge-Kutta method is used. If we don't
# provide the time step explicitly, an adaptive time step is used.
tlims = (T(0), T(10))
state, outputs = solve_unsteady(setup, u₀, tlims; Δt = T(1e-3), psolver, processors);

# ## Post-process
#
# We may visualize or export the computed fields

# Export fields to VTK. The file `output/solution.vti` may be opened for
# visualization in [ParaView](https://www.paraview.org/). This is particularly
# useful for inspecting results from 3D simulations.
save_vtk(setup, state.u, state.t, "$output/solution"; psolver)

# Plot pressure
fieldplot(state; setup, fieldname = :pressure)

# Plot velocity. Note the time stamp used for computing boundary conditions, if
# any.
fieldplot(state; setup, fieldname = :velocity)

# Plot vorticity
fieldplot(state; setup, fieldname = :vorticity)

# In addition, the named tuple `outputs` contains quantities from our
# processors.
# The logger returns nothing.

## outputs.rtp
## outputs.ehist
## outputs.espec
## outputs.anim 
## outputs.vtk
## outputs.field
outputs.log
