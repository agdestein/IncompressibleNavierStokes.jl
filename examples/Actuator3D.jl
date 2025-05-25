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
using WGLMakie #!md
using IncompressibleNavierStokes
## using CUDA

# Output directory
outdir = joinpath(@__DIR__, "output", "Actuator3D")

# Floating point type
T = Float64

# Reynolds number

# A 3D grid is a Cartesian product of three vectors
x = LinRange(0.0, 6.0, 31), LinRange(-2.0, 2.0, 41), LinRange(-2.0, 2.0, 41)
plotgrid(x...)

# Boundary conditions: Unsteady BC requires time derivatives
boundary_conditions = (;
    u = (
        ## x left, x right
        (
            DirichletBC(
                (dim, x, y, z, t) ->
                    dim == 1 ? cos(π / 6 * sin(π / 6 * t)) :
                    dim == 2 ? sin(π / 6 * sin(π / 6 * t)) : zero(x),
            ),
            PressureBC(),
        ),

        ## y rear, y front
        (PressureBC(), PressureBC()),

        ## z rear, z front
        (PressureBC(), PressureBC()),
    )
)

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(f, state, t; setup, cache, viscosity)
    navierstokes!(f, state, t; setup, cache, viscosity)
    f.u .+= cache.bodyforce
end

# Tell IncompressibleNavierStokes how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
function IncompressibleNavierStokes.get_cache(::typeof(force!), setup)
    # Actuator body force: A thrust coefficient `Cₜ` distributed over a short cylinder
    cx, cy, cz = T(2), T(0), T(0) # Disk center
    D = T(1)                      # Disk diameter
    δ = T(0.11)                   # Disk thickness
    Cₜ = T(0.2)                  # Thrust coefficient
    cₜ = Cₜ / (π * (D / 2)^2 * δ)
    inside(x, y, z) = abs(x - cx) ≤ δ / 2 && (y - cy)^2 + (z - cz)^2 ≤ (D / 2)^2
    f(dim, x, y, z) = dim == 1 ? -cₜ * inside(x, y, z) : zero(x)
    bodyforce = velocityfield(setup, f; doproject = false)
    (; bodyforce)
end

# We also need to tell how to propos the time step sizes for our given force.
# We just fall back to the default one.
IncompressibleNavierStokes.propose_timestep(::typeof(force!), state, setup, params) =
    IncompressibleNavierStokes.propose_timestep(navierstokes!, state, setup, params)

# Build setup and assemble operators
setup = Setup(;
    x,
    boundary_conditions,
    ## backend = CUDABackend(),
);

# This will factorize the Laplace matrix
psolver = default_psolver(setup);

# Initial conditions (extend inflow)
u = velocityfield(setup, (dim, x, y, z) -> dim == 1 ? one(x) : zero(x); psolver);

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    force!,
    start = (; u),
    tlims = (T(0), T(3)),
    params = (; viscosity = T(1e-2),),
    psolver,
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = energy_history_plot,
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
save_vtk(state; setup, filename = joinpath(outdir, "solution"))
