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

# Choose backend
backend = IncompressibleNavierStokes.CPU()
## using CUDA; backend = CUDABackend()

# Boundary conditions
boundary_conditions = (
    ## x left, x right
    (DirichletBC(), DirichletBC()),

    ## y bottom, y top
    (DirichletBC(), DirichletBC((1.0, 0.0))),
)

# We create a two-dimensional domain with a box of size `[1, 1]`. The grid is
# created as a Cartesian product between two vectors. We add a refinement near
# the walls.
n = 32
ax = tanh_grid(0.0, 1.0, n)
plotgrid(ax, ax)

# We can now build the setup and assemble operators.
# A 3D setup is built if we also provide a vector of z-coordinates.
setup = Setup(; x = (ax, ax), boundary_conditions, Re = 1e3, backend);

# Initial conditions
u = velocityfield(setup, (dim, x, y) -> zero(x));

# Iteration processors are called after every `nupdate` time steps. This can be
# useful for logging, plotting, or saving results. Their respective outputs are
# later returned by `solve_unsteady`.

processors = (
    ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 50),
    ## ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 10),
    ## espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
    ## anim = animator(; setup, path = "$outdir/solution.mkv", nupdate = 20),
    ## vtk = vtk_writer(; setup, nupdate = 100, dir = outdir, filename = "solution"),
    ## field = fieldsaver(; setup, nupdate = 10),
    log = timelogger(; nupdate = 1000),
);

state, outputs = solve_unsteady(; setup, start = (; u), tlims = (0.0, 10.0), processors);

# ## Post-process
#
# We may visualize or export the computed fields

# Export fields to VTK. The file `outdir/solution.vti` may be opened for
# visualization in [ParaView](https://www.paraview.org/). This is particularly
# useful for inspecting results from 3D simulations.

filenmae = joinpath(@__DIR__, "output", "solution")
## save_vtk(state; setup, filename)

# Plot velocity
fieldplot(state; setup, fieldname = :velocitynorm)

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
