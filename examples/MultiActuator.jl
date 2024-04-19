# # Unsteady actuator case - 2D
#
# In this example, an unsteady inlet velocity profile at encounters a wind
# turbine blade in a wall-less domain. The blade is modeled as a uniform body
# force on a thin rectangle.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using Adapt
using Random

# Output directory
output = "output/Actuator2D"

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using CUDA;
T = Float32;
# T = Float64;
ArrayType = CuArray;
CUDA.allowscalar(false);
device = x -> adapt(ArrayType, x);

set_theme!(; GLMakie = (; scalefactor = 1.5))

const ππ = T(π)

# Boundary conditions
boundary_conditions = (
    ## x left, x right
    (
        ## Unsteady BC requires time derivatives
        DirichletBC(
            (dim, x, y, t) -> sin(ππ / 6 * sin(ππ / 6 * t) + ππ / 2 * (dim() == 1)),
            (dim, x, y, t) ->
                (ππ / 6)^2 *
                cos(ππ / 6 * t) *
                cos(ππ / 6 * sin(ππ / 6 * t) + ππ / 2 * (dim() == 1)),
        ),
        PressureBC(),
    ),

    ## y rear, y front
    (PressureBC(), PressureBC()),
)

# Actuator body force: A thrust coefficient `Cₜ` distributed over a thin rectangle
create_bodyforce(; xc, yc, D, δ, C) =
    (dim, x, y, t) ->
        dim() == 1 && abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2 ? -C / (D * δ) : zero(x)

create_manyforce(forces...) = function (dim, x, y, t)
    out = zero(x)
    for f in forces
        out += f(dim, x, y, t)
    end
    out
end

bodyforce = create_bodyforce(;
    xc = T(2),   # Disk center
    yc = T(0),   # Disk center
    D = T(1),    # Disk diameter
    δ = T(0.11), # Disk thickness
    C = T(0.2),  # Thrust coefficient
)

disk = (; D = T(1), δ = T(0.11), C = T(0.2))
bodyforce = create_manyforce(
    create_bodyforce(; xc = T(2), yc = T(0), disk...),
    create_bodyforce(; xc = T(4), yc = T(0.7), disk...),
    create_bodyforce(; xc = T(6.4), yc = T(-1), disk...),
)

# A 2D grid is a Cartesian product of two vectors
n = 160
x = LinRange(T(0), T(10), 5n + 1)
y = LinRange(-T(2), T(2), 2n + 1)

# Build setup and assemble operators
setup = Setup(x, y; Re = T(2000), boundary_conditions, bodyforce, ArrayType);

using CUDA.CUSPARSE
using CUDSS
using LinearAlgebra

psolver = CUDSSPressureSolver(setup);
psolver2 = DirectPressureSolver(setup);

p = fill!(similar(setup.grid.x[1], setup.grid.N), 0)
f = similar(setup.grid.x[1], setup.grid.N)
f .= randn.(T)
IncompressibleNavierStokes.apply_bc_p!(f, T(0), setup)

IncompressibleNavierStokes.poisson!(psolver, p, f)
IncompressibleNavierStokes.poisson!(psolver2, p, f)

using KernelAbstractions
using BenchmarkTools

@benchmark begin
    # IncompressibleNavierStokes.poisson!(psolver, p, f)
    IncompressibleNavierStokes.poisson!(psolver2, p, f)
    KernelAbstractions.synchronize(get_backend(p))
end

# Initial conditions (extend inflow)
u₀ = create_initial_conditions(setup, (dim, x, y) -> dim() == 1 ? one(x) : zero(x); psolver);
u = u₀
t = T(0)

# Solve unsteady problem
state, outputs = solve_unsteady(
    setup,
    u₀,
    (T(0), 4 * T(12));
    # (T(0), T(1));
    method = RK44P2(),
    Δt = T(0.01),
    psolver,
    processors = (
        rtp = realtimeplotter(;
            setup,
            # plot = fieldplot,
            # fieldname = :velocity,
            # fieldname = :pressure,
            psolver,
            nupdate = 1,
        ),
        boxplotter = processor() do state
            for box in boxes
                lines!(current_axis(), box; color = :red);
            end
        end,
        ## ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 1),
        ## espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 1),
        ## anim = animator(; setup, path = "$output/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = "$output", filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`.

# Export to VTK
save_vtk(setup, state.u, state.p, "$output/solution")

# We create a box to visualize the actuator.
(; xc, yc, D, δ) = setup.bodyforce
box = [
    Point2f(xc - δ / 2, yc + D / 2),
    Point2f(xc - δ / 2, yc - D / 2),
    Point2f(xc + δ / 2, yc - D / 2),
    Point2f(xc + δ / 2, yc + D / 2),
    Point2f(xc - δ / 2, yc + D / 2),
]

box = boxes[1]
boxes = map(bodyforce.forces) do (; xc, yc, D, δ)
    [
        Point2f(xc - δ / 2, yc + D / 2),
        Point2f(xc - δ / 2, yc - D / 2),
        Point2f(xc + δ / 2, yc - D / 2),
        Point2f(xc + δ / 2, yc + D / 2),
        Point2f(xc - δ / 2, yc + D / 2),
    ]
end

state = (; u = F, t)

# Plot pressure
fig = fieldplot(state; setup, fieldname = :pressure, psolver)
# lines!(box...; color = :red)
lines!.(boxes; color = :red);
fig

sum(IncompressibleNavierStokes.pressure(u, t, setup; psolver))

# Plot velocity
fig = fieldplot(state; setup, fieldname = :velocity)
# lines!(box...; color = :red)
lines!.(boxes; color = :red);
fig

# Plot vorticity
fig = fieldplot(state; setup, fieldname = :vorticity)
# lines!(box...; color = :red)
lines!.(boxes; color = :red);
fig
