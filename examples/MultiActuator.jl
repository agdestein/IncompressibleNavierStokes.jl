# # Unsteady actuator case - 2D
#
# In this example, an unsteady inlet velocity profile at encounters a wind
# turbine blade in a wall-less domain. The blade is modeled as a uniform body
# force on a thin rectangle.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using Random

# Output directory
outdir = joinpath(@__DIR__, "output", "MultiActuator")

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Boundary conditions
boundary_conditions = (
    ## x left, x right
    (
        DirichletBC((dim, x, y, t) -> sinpi(sinpi(t / 6) / 6 + one(x) / 2 * (dim == 1))),
        PressureBC(),
    ),

    ## y rear, y front
    (PressureBC(), PressureBC()),
)

# Actuator body force: A thrust coefficient `Cₜ` distributed over a thin rectangle
create_bodyforce(; xc, yc, D, δ, C) =
    (dim, x, y, t) ->
        dim == 1 && abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2 ? -C / (D * δ) : zero(x)

create_manyforce(forces...) = function (dim, x, y, t)
    out = zero(x)
    for f in forces
        out += f(dim, x, y, t)
    end
    out
end

disk = (; D = T(1), δ = T(0.11), C = T(0.2))
bodyforce = create_manyforce(
    create_bodyforce(; xc = T(2), yc = T(0), disk...),
    create_bodyforce(; xc = T(4), yc = T(0.7), disk...),
    create_bodyforce(; xc = T(6.4), yc = T(-1), disk...),
)

# A 2D grid is a Cartesian product of two vectors
n = 50
x = LinRange(T(0), T(10), 5n + 1), LinRange(-T(2), T(2), 2n + 1)
plotgrid(x...; figure = (; size = (600, 300)))

# Build setup and assemble operators
setup = Setup(;
    x,
    Re = T(1000),
    boundary_conditions,
    bodyforce,
    issteadybodyforce = true,
    ArrayType,
);

# Initial conditions (extend inflow)
ustart = velocityfield(setup, (dim, x, y) -> dim == 1 ? one(x) : zero(x));
t = T(0)

boxes = map(bodyforce.forces) do (; xc, yc, D, δ)
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
    ustart,
    tlims = (T(0), T(12)),
    method = RKMethods.RK44P2(),
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

# Plot pressure
fig = fieldplot(state; setup, size = (600, 300), fieldname = :pressure)
lines!.(boxes; color = :red);
fig

# Plot velocity
fig = fieldplot(state; setup, size = (600, 300), fieldname = :velocitynorm)
lines!.(boxes; color = :red);
fig

# Plot vorticity
fig = fieldplot(state; setup, size = (600, 300), fieldname = :vorticity)
lines!.(boxes; color = :red);
fig
