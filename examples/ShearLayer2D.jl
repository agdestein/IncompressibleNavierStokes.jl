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

# Output directory
outdir = joinpath(@__DIR__, "output", "ShearLayer2D")

# Floating point type
T = Float64

# Backend
backend = CPU()
## using CUDA; backend = CUDABackend()

# Reynolds number
Re = T(2000)

# A 2D grid is a Cartesian product of two vectors
n = 128
lims = T(0), T(2π)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
plotgrid(x...)

# Build setup and assemble operators
setup = Setup(; x, Re, backend);

# Initial conditions: We add 1 to u in order to make global momentum
# conservation less trivial
d = T(π / 15)
e = T(0.05)
U1(y) = y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d)
## U1(y) = T(1) + (y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d))
ustart = velocityfield(setup, (dim, x, y) -> dim == 1 ? U1(y) : e * sin(x));

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(8)),
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
);

# ## Post-process
#
# We may visualize or export the computed fields

outputs.rtp

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
