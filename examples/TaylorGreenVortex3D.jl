# # Taylor-Green vortex - 3D
#
# In this example we consider the Taylor-Green vortex.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Floating point precision
T = Float64

# ## Array type
#
# Running in 3D is heavier than in 2D.
# If you are running this on a CPU, consider using multiple threads by
# starting Julia with `julia -t auto`, or
# add `-t auto` to # the `julia.additionalArgs` # setting in VSCode.

ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# ## Setup

n = 32
r = range(T(0), T(1), n + 1)
setup = Setup(; x = (r, r, r), Re = T(1e3), ArrayType);
psolver = psolver_spectral(setup);

# Initial conditions
U(dim, x, y, z) =
    if dim == 1
        sinpi(2x) * cospi(2y) * sinpi(2z) / 2
    elseif dim == 2
        -cospi(2x) * sinpi(2y) * sinpi(2z) / 2
    else
        zero(x)
    end
ustart = velocityfield(setup, U, psolver);

# ## Solve unsteady problem

state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(1.0)),
    Δt = T(1e-3),
    processors = (
        ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 10),
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 1,
            displayfig = false,
        ),
        espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        ## anim = animator(; setup, path = "$outdir/solution.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = outdir, filename = "solution"),
        log = timelogger(; nupdate = 100),
    ),
    psolver,
);

# ## Post-process

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec
