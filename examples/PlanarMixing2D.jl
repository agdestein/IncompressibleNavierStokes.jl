# # Planar mixing - 2D
#
# Planar mixing example, as presented in [List2022](@cite).

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory
output = "output/PlanarMixing2D"

# Viscosity model
Re = 500.0

# Boundary conditions: Unsteady BC requires time derivatives
ΔU = 1.0
Ubar = 1.0
ϵ = (0.082Ubar, 0.012Ubar)
n = (0.4π, 0.3π)
ω = (0.22, 0.11)
U(dim, x, y, t) =
    dim() == 1 ?
    1.0 + ΔU / 2 * tanh(2y) + sum(@. ϵ * (1 - tanh(y / 2)^2) * cos(n * y) * sin(ω * t)) :
    0.0
dUdt(dim, x, y, t) =
    dim() == 1 ? sum(@. ϵ * (1 - tanh(y / 2)^2) * cos(n * y) * ω * cos(ω * t)) : 0.0
boundary_conditions = (
    ## x left, x right
    (DirichletBC(U, dUdt), PressureBC()),

    ## y rear, y front
    (PressureBC(), PressureBC()),
)

# A 2D grid is a Cartesian product of two vectors
n = 64
## n = 256
x = LinRange(0.0, 256.0, 4n)
y = LinRange(-32.0, 32.0, n)
plotgrid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; Re, boundary_conditions);
psolver = psolver_direct(setup);

# Initial conditions (extend inflow)
u₀ = create_initial_conditions(setup, (dim, x, y) -> U(dim, x, y, 0.0); psolver);

# Solve unsteady problem
state, outputs = solve_unsteady(
    setup,
    u₀,
    (0.0, 100.0);
    psolver,
    method = RKMethods.RK44P2(),
    Δt = 0.1,
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            ## plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            nupdate = 1,
        ),
        ## anim = animator(; setup, path = "$output/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = output, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`

outputs.rtp

# Export to VTK
save_vtk(setup, state.u, state.t, "$output/solution"; psolver)
