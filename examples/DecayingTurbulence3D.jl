# # Decaying Homogeneous Isotropic Turbulence - 3D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

#md using CairoMakie
using WGLMakie #!md
using IncompressibleNavierStokes
## using CUDA; 

# ## Problem setup

T = Float32
n = 128
ax = range(T(0), T(1), n + 1)
setup = Setup(;
    x = (ax, ax, ax),
    boundary_conditions = (;
        u = (
            (PeriodicBC(), PeriodicBC()),
            (PeriodicBC(), PeriodicBC()),
            (PeriodicBC(), PeriodicBC()),
        ),
    ),
    backend = CUDABackend(),
);
psolver = default_psolver(setup);
u = random_field(setup; psolver);

# ## Solve problem

state, outputs = solve_unsteady(;
    setup,
    start = (; u),
    tlims = (T(0), T(1)),
    psolver,
    params = (; viscosity = T(2e-4)),
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
