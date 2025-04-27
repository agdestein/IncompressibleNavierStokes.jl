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

# Output directory
outdir = joinpath(@__DIR__, "output", "BackwardFacingStep3D")

# Floating point type
T = Float32

# Backend
backend = IncompressibleNavierStokes.CPU()
## using CUDA; backend = CUDABackend()

# Reynolds number
Re = T(1000)

# A 3D grid is a Cartesian product of three vectors
x = LinRange(T(0), T(10), 129),
LinRange(-T(0.5), T(0.5), 17),
LinRange(-T(0.25), T(0.25), 9)
plotgrid(x...)

# Boundary conditions: steady inflow on the top half
U(dim, x, y, z, t) = dim == 1 && y ≥ 0 ? 24y * (one(x) / 2 - y) : zero(x)
boundary_conditions = (
    ## x left, x right
    (DirichletBC(U), PressureBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC()),

    ## z bottom, z top
    (PeriodicBC(), PeriodicBC()),
)

# Build setup and assemble operators
setup = Setup(; x, Re, boundary_conditions, backend);

# Initial conditions (extend inflow)
ustart = velocityfield(setup, (dim, x, y, z) -> U(dim, x, y, z, zero(x)));

# Solve steady state problem
## u, p = solve_steady_state(setup, u₀, p₀);
nothing

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(7)),
    Δt = T(0.01),
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            ## plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            nupdate = 1,
        ),
        ## anim = animator(; setup, path = "$outdir/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = outdir, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
)

# ## Post-process
#
# We may visualize or export the computed fields

# Export to VTK
save_vtk(state; setup, filename = joinpath(outdir, "solution"))

# Plot pressure
fieldplot(state; setup, fieldname = :pressure)

# Plot velocity
fieldplot(state; setup, fieldname = :velocitynorm)

# Plot vorticity
fieldplot(state; setup, fieldname = :vorticity)

#md # ## Copy-pasteable code
#md #
#md # Below is the full code for this example stripped of comments and output.
#md #
#md # ```julia
#md # CODE_CONTENT
#md # ```
