# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Backward Facing Step - 3D
#
# In this example we consider a channel with periodic side boundaries, walls at
# the top and bottom, and a step at the left with a parabolic inflow. Initially
# the velocity is an extension of the inflow, but as time passes the velocity
# finds a new steady state.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory
output = "output/BackwardFacingStep3D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(3000)

# A 3D grid is a Cartesian product of three vectors
x = LinRange(T(0), T(10), 129)
y = LinRange(-T(0.5), T(0.5), 17)
z = LinRange(-T(0.25), T(0.25), 9)
plotgrid(x, y, z)

# Boundary conditions: steady inflow on the top half
U(dim, x, y, z, t) = dim() == 1 && y ≥ 0 ? 24y * (one(x) / 2 - y) : zero(x)
dUdt(dim, x, y, z, t) = zero(x)
boundary_conditions = (
    ## x left, x right
    (DirichletBC(U, dUdt), PressureBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC()),

    ## z bottom, z top
    (PeriodicBC(), PeriodicBC()),
)

# Build setup and assemble operators
setup = Setup(x, y, z; Re, boundary_conditions, ArrayType);

# Initial conditions (extend inflow)
u₀ = create_initial_conditions(setup, (dim, x, y, z) -> U(dim, x, y, z, zero(x)));

# Solve steady state problem
## u, p = solve_steady_state(setup, u₀, p₀);
nothing

# Solve unsteady problem
state, outputs = solve_unsteady(
    setup,
    u₀,
    (T(0), T(7));
    Δt = T(0.01),
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
)

# ## Post-process
#
# We may visualize or export the computed fields

# Export to VTK
save_vtk(setup, state.u, state.p, "$output/solution")

# Plot pressure
fieldplot(state; setup, fieldname = :pressure)

# Plot velocity
fieldplot(state; setup, fieldname = :velocity)

# Plot vorticity
fieldplot(state; setup, fieldname = :vorticity)
