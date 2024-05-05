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
output = "output/DecayingTurbulence2D"

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
u₀ = random_field(setup, T(0));

# Solve unsteady problem
state, outputs = solve_unsteady(
    setup,
    u₀,
    (T(0), T(1));
    Δt = T(1e-3),
    processors = (
        ## rtp = realtimeplotter(; setup, nupdate = 1),
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 10,
            displayfig = false,
        ),
        espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        ## anim = animator(; setup, path = "$output/solution.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = output, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec

# Export to VTK
save_vtk(setup, state.u, state.t, "$output/solution")

# Plot field
fieldplot(state; setup)
