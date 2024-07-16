# # Unsteady actuator case - 3D
#
# In this example, an unsteady inlet velocity profile at encounters a wind
# turbine blade in a wall-less domain. The blade is modeled as a uniform body
# force on a short cylinder.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory
outdir = joinpath(@__DIR__, "output", "Actuator3D")

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(100)

# A 3D grid is a Cartesian product of three vectors
x = LinRange(0.0, 6.0, 31)
y = LinRange(-2.0, 2.0, 41)
z = LinRange(-2.0, 2.0, 41)
plotgrid(x, y, z)

# Boundary conditions: Unsteady BC requires time derivatives
boundary_conditions = (
    ## x left, x right
    (
        DirichletBC(
            (dim, x, y, z, t) ->
                dim() == 1 ? cos(π / 6 * sin(π / 6 * t)) :
                dim() == 2 ? sin(π / 6 * sin(π / 6 * t)) : zero(x),
            (dim, x, y, z, t) ->
                dim() == 1 ? -(π / 6)^2 * cos(π / 6 * t) * sin(π / 6 * sin(π / 6 * t)) :
                dim() == 2 ? (π / 6)^2 * cos(π / 6 * t) * cos(π / 6 * sin(π / 6 * t)) :
                zero(x),
        ),
        PressureBC(),
    ),

    ## y rear, y front
    (PressureBC(), PressureBC()),

    ## z rear, z front
    (PressureBC(), PressureBC()),
)

# Actuator body force: A thrust coefficient `Cₜ` distributed over a short cylinder
cx, cy, cz = T(2), T(0), T(0) # Disk center
D = T(1)                      # Disk diameter
δ = T(0.11)                   # Disk thickness
Cₜ = T(0.2)                  # Thrust coefficient
cₜ = Cₜ / (π * (D / 2)^2 * δ)
inside(x, y, z) = abs(x - cx) ≤ δ / 2 && (y - cy)^2 + (z - cz)^2 ≤ (D / 2)^2
bodyforce(dim, x, y, z) = dim() == 1 ? -cₜ * inside(x, y, z) : zero(x)

# Build setup and assemble operators
setup = Setup(x, y, z; Re, boundary_conditions, bodyforce, ArrayType);

# Initial conditions (extend inflow)
ustart = create_initial_conditions(setup, (dim, x, y, z) -> dim() == 1 ? one(x) : zero(x));

# Solve unsteady problem
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(3)),
    method = RKMethods.RK44P2(),
    Δt = T(0.05),
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            ## plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            nupdate = 1,
        ),
        ## anim = animator(; setup, path = "$outdir/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = "$outdir", filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Field plot
outputs.rtp

# Export to VTK
save_vtk(setup, u, t, "$outdir/solution")
