# # Unsteady actuator case - 2D
#
# In this example, an unsteady inlet velocity profile at encounters a wind
# turbine blade in a wall-less domain. The blade is modeled as a uniform body
# force on a thin rectangle.

# ## Packages
#
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# ## Setup

# A 2D grid is a Cartesian product of two vectors
n = 40
x = LinRange(0.0, 10.0, 5n + 1), LinRange(-2.0, 2.0, 2n + 1)
plotgrid(x...; figure = (; size = (600, 300)))

# Boundary conditions
inflow(dim, x, y, t) = sinpi(sinpi(t / 6) / 6 + (dim == 1) / 2)
boundary_conditions = ((DirichletBC(inflow), PressureBC()), (PressureBC(), PressureBC()))

# Build setup
setup = Setup(; x, visc = 1e-2, boundary_conditions);

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(f, state, t, params, setup, cache)
    navierstokes!(f, state, t, nothing, setup, nothing)
    f.u .+= cache.bodyforce
end

# Actuator body force: A thrust coefficient `Cₜ` distributed over a thin rectangle
xc, yc = 2.0, 0.0 # Disk center
D = 1.0           # Disk diameter
δ = 0.11          # Disk thickness
C = 0.2           # Thrust coefficient
c = C / (D * δ)   # Normalize
inside(x, y) = abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2
f(dim, x, y) = -c * (dim == 1) * inside(x, y)

# Tell IncompressibleNavierStokes how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
function IncompressibleNavierStokes.get_cache(::typeof(force!), setup)
    bodyforce = velocityfield(setup, f; doproject = false)
    (; bodyforce)
end

# Initial conditions (extend inflow)
u = velocityfield(setup, (dim, x, y) -> inflow(dim, x, y, 0.0))

# ## Solve unsteady problem

state, outputs = solve_unsteady(;
    setup,
    force!,
    start = (; u),
    tlims = (0.0, 12.0),
    processors = (
        rtp = realtimeplotter(; setup, size = (600, 300), nupdate = 5),
        log = timelogger(; nupdate = 50),
    ),
);

#md # ```@raw html
#md # <video src="/Actuator2D.mp4" controls="controls" autoplay="autoplay" loop="loop"></video>
#md # ```

# ## Post-process

# We create a box to visualize the actuator.
box = (
    [xc - δ / 2, xc - δ / 2, xc + δ / 2, xc + δ / 2, xc - δ / 2],
    [yc + D / 2, yc - D / 2, yc - D / 2, yc + D / 2, yc + D / 2],
)

# Plot velocity
fig = fieldplot(state; setup, size = (600, 300), fieldname = :velocitynorm)
lines!(box...; color = :red)
fig

# Plot vorticity
fig = fieldplot(state; setup, size = (600, 300), fieldname = :vorticity)
lines!(box...; color = :red)
fig
