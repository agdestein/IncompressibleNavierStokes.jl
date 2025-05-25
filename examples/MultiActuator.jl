# # Unsteady actuator case - 2D
#
# In this example, an unsteady inlet velocity profile at encounters a wind
# turbine blade in a wall-less domain. The blade is modeled as a uniform body
# force on a thin rectangle.

#md using CairoMakie
using WGLMakie #!md
using IncompressibleNavierStokes
using Random

# Output directory
outdir = joinpath(@__DIR__, "output", "MultiActuator")

# Boundary conditions
boundary_conditions = (;
    u = (
        ## x left, x right
        (
            DirichletBC(
                (dim, x, y, t) -> sinpi(sinpi(t / 6) / 6 + one(x) / 2 * (dim == 1)),
            ),
            PressureBC(),
        ),

        ## y rear, y front
        (PressureBC(), PressureBC()),
    )
)

# Actuator body force: A thrust coefficient `Cₜ` distributed over a thin rectangle
create_bodyforce(; xc, yc, D, δ, C) =
    (dim, x, y) ->
        dim == 1 && abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2 ? -C / (D * δ) : zero(x)

create_manyforce(forces...) = function (dim, x, y)
    out = zero(x)
    for f in forces
        out += f(dim, x, y)
    end
    out
end

disk = (; D = T(1), δ = T(0.11), C = T(0.2))
f = create_manyforce(
    create_bodyforce(; xc = T(2), yc = T(0), disk...),
    create_bodyforce(; xc = T(4), yc = T(0.7), disk...),
    create_bodyforce(; xc = T(6.4), yc = T(-1), disk...),
)

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(f, state, t, params, setup, cache)
    navierstokes!(f, state, t, nothing, setup, nothing)
    f.u .+= cache.bodyforce
end

# Tell IncompressibleNavierStokes how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
function IncompressibleNavierStokes.get_cache(::typeof(force!), setup)
    bodyforce = velocityfield(setup, f; doproject = false)
    (; bodyforce)
end

# We also need to tell how to propos the time step sizes for our given force.
# We just fall back to the default one.
IncompressibleNavierStokes.propose_timestep(::typeof(force!), state, setup, params) =
    IncompressibleNavierStokes.propose_timestep(navierstokes!, state, setup, params)

# A 2D grid is a Cartesian product of two vectors
n = 50
x = LinRange(0.0, 10.0, 5n + 1), LinRange(-2.0, 2.0, 2n + 1)
plotgrid(x...; figure = (; size = (600, 300)))

# Build setup and assemble operators
setup = Setup(; x, boundary_conditions);

# Initial conditions (extend inflow)
u = velocityfield(setup, (dim, x, y) -> (dim == 1) * one(x));

boxes = map(f.forces) do (; xc, yc, D, δ)
    [
        Point2f(xc - δ / 2, yc + D / 2),
        Point2f(xc - δ / 2, yc - D / 2),
        Point2f(xc + δ / 2, yc - D / 2),
        Point2f(xc + δ / 2, yc + D / 2),
        Point2f(xc - δ / 2, yc + D / 2),
    ]
end
box = boxes[1]

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    force!,
    start = (; u),
    tlims = (0.0, 12.0),
    params = (; viscosity = 5e-3),
    processors = (
        rtp = realtimeplotter(;
            setup,
            ## plot = fieldplot,
            ## fieldname = :velocitynorm,
            ## fieldname = :pressure,
            size = (600, 300),
            nupdate = 1,
        ),
        boxplotter = processor() do state
            for box in boxes
                lines!(current_axis(), box; color = :red)
            end
        end,
        ## ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 1),
        ## espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 1),
        ## anim = animator(; setup, path = "$outdir/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = "$outdir", filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`.

# Export to VTK
save_vtk(state; setup, filename = joinpath(outdir, "solution"))

# Plot velocity
fig = fieldplot(state; setup, size = (600, 300), fieldname = :velocitynorm)
lines!.(boxes; color = :red);
fig

# Plot vorticity
fig = fieldplot(state; setup, size = (600, 300), fieldname = :vorticity)
lines!.(boxes; color = :red);
fig
