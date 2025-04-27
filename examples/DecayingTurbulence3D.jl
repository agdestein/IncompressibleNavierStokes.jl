# # Decaying Homogeneous Isotropic Turbulence - 3D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# ## Problem setup

backend = IncompressibleNavierStokes.CPU()
## using CUDA; backend = CUDABackend()
T = Float32
n = 128
ax = range(T(0), T(1), n + 1)
setup = Setup(; x = (ax, ax, ax), Re = T(4e3), backend);
psolver = default_psolver(setup);
ustart = random_field(setup; psolver);

# ## Solve problem

state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(2)),
    psolver,
    processors = (
        ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 10),
        ## ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 10),
        espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        ## anim = animator(; setup, path = "$outdir/solution.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = outdir, filename = "solution"),
        log = timelogger(; nupdate = 10),
    ),
);

# ## Post-process

# Field plot
## outputs.rtp
## fieldplot(state; setup, levels = 0:5)

# Energy history
## outputs.ehist

# Energy spectrum
outputs.espec

#md # ## Copy-pasteable code
#md #
#md # Below is the full code for this example stripped of comments and output.
#md #
#md # ```julia
#md # CODE_CONTENT
#md # ```
