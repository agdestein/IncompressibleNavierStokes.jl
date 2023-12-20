# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Shear layer - 2D
#
# Shear layer example.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory
output = "output/ShearLayer2D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(Inf)

# A 2D grid is a Cartesian product of two vectors
n = 128
lims = T(0), T(2π)
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
plotgrid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; Re, ArrayType);

psolver = SpectralPressureSolver(setup)

# Initial conditions: We add 1 to u in order to make global momentum
# conservation less trivial
d = T(π / 15)
e = T(0.05)
U1(y) = y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d)
## U1(y) = T(1) + (y ≤ π ? tanh((y - T(π / 2)) / d) : tanh((T(3π / 2) - y) / d))
u₀ = create_initial_conditions(
    setup,
    (dim, x, y) -> dim() == 1 ? U1(y) : e * sin(x);
    psolver,
);

# Solve unsteady problem
state, outputs = solve_unsteady(
    setup,
    u₀,
    (T(0), T(8));
    Δt = T(0.01),
    psolver,
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
# We may visualize or export the computed fields

outputs.rtp

# Export to VTK
save_vtk(setup, state.u, state.p, "$output/solution")

# Plot pressure
fieldplot(state; setup, fieldname = :pressure)

# Plot velocity
fieldplot(state; setup, fieldname = :velocity)

# Plot vorticity
fieldplot(state; setup, fieldname = :vorticity)
