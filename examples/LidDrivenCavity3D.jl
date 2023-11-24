# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Lid-Driven Cavity - 3D
#
# In this example we consider a box with a moving lid. The velocity is initially at rest. The
# solution should reach at steady state equilibrium after a certain time. The same steady
# state should be obtained when solving a steady state problem.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "LidDrivenCavity3D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(1_000)

# A 3D grid is a Cartesian product of three vectors. Here we refine the grid
# near the walls.
x = cosine_grid(T(0), T(1), 25)
y = cosine_grid(T(0), T(1), 25)
z = LinRange(-T(0.2), T(0.2), 11)
plot_grid(x, y, z)

# Boundary conditions: horizontal movement of the top lid
U(dim, x, y, z, t) = dim() == 1 ? one(x) : dim() == 2 ? zero(x) : one(x) / 5
dUdt(dim, x, y, z, t) = zero(x)
boundary_conditions = (
    ## x left, x right
    (DirichletBC(), DirichletBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC(U, dUdt)),

    ## z bottom, z top
    (PeriodicBC(), PeriodicBC()),
)

# Build setup and assemble operators
setup = Setup(x, y, z; Re, boundary_conditions, ArrayType);

# Initial conditions
u₀, p₀ = create_initial_conditions(setup, (dim, x, y, z) -> zero(x))

# Solve steady state problem
## u, p = solve_steady_state(setup, u₀, p₀; npicard = 5, maxiter = 15);
nothing

# Solve unsteady problem
u, p, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    (T(0), T(0.2));
    Δt = T(0.001),
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            ## plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            nupdate = 1,
        ),
        ## anim = animator(; setup, path = "vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = "output/$name", filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(setup, u, p, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, u)

# Plot vorticity
plot_vorticity(setup, u)

# Plot streamfunction
## plot_streamfunction(setup, u)
nothing
