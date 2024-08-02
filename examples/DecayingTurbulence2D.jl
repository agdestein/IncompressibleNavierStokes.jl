# # Decaying Homogeneous Isotropic Turbulence - 2D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory
outdir = joinpath(@__DIR__, "output", "DecayingTurbulence2D")

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Viscosity model
Re = T(10_000)

# A 2D grid is a Cartesian product of two vectors
n = 256
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)

# Build setup and assemble operators
setup = Setup(x...; Re, ArrayType);

# Create random initial conditions
ustart = random_field(setup, T(0));

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(1)),
    Δt = T(1e-3),
    processors = (
        rtp = realtimeplotter(; setup, nupdate = 10),
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 10,
            displayfig = false,
        ),
        espec = realtimeplotter(;
            setup,
            plot = energy_spectrum_plot,
            nupdate = 10,
            displayfig = false,
        ),
        ## anim = animator(; setup, path = joinpath(outdir, "solution.mp4"), nupdate = 10),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = outdir, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

#md # ```@raw html
#md # <video src="/DecayingTurbulence2D.mp4" controls="controls" autoplay="autoplay" loop="loop"></video>
#md # ```

# ## Post-process
#
# We may visualize or export the computed fields

# Export to VTK
save_vtk(state; setup, filename = joinpath(outdir, "solution"))

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec

# Plot field
fieldplot(state; setup)
