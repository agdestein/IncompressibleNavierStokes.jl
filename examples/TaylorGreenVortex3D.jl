# # Taylor-Green vortex - 3D
#
# In this example we consider the Taylor-Green vortex.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory
outdir = joinpath(@__DIR__, "output", "TaylorGreenVortex3D")
ispath(outdir) || mkpath(outdir)

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(6_000)

# A 3D grid is a Cartesian product of three vectors
n = 32
lims = T(0), T(1)
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
z = LinRange(lims..., n + 1)

# Build setup and assemble operators
setup = Setup(x, y, z; Re, ArrayType);

# Since the grid is uniform and identical for x, y, and z, we may use a
# specialized spectral pressure solver
psolver = psolver_spectral(setup);

# Initial conditions
ustart = create_initial_conditions(
    setup,
    (dim, x, y, z) ->
        dim() == 1 ? sinpi(2x) * cospi(2y) * sinpi(2z) / 2 :
        dim() == 2 ? -cospi(2x) * sinpi(2y) * sinpi(2z) / 2 : zero(x);
    psolver,
);

# Solve unsteady problem
(; u, t), outputs = solve_unsteady(;
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
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
    psolver,
);

# ## Post-process
#
# We may visualize or export the computed fields

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec

# Export to VTK
save_vtk(state; setup, filename = joinpath(outdir, "solution"), psolver)
