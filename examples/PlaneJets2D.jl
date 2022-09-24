# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Plane jets - 2D
#
# Plane jets example, as presented in [MacArt2021](@cite). Note that the
# original formulation is in 3D.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

using FFTW
#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using LaTeXStrings

# Case name for saving results
name = "PlaneJets2D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 6000.0)

# Test cases (A, B, C, D; in order)
# U() = sqrt(467.4)
U() = 21.619435700313733

u_A(y) = U() / 2 * (tanh((y + 1 / 2) / 0.1) - tanh((y - 1 / 2) / 0.1))

u_B(y) =
    U() / 2 * (tanh((y + 1 + 1 / 2) / 0.1) - tanh((y + 1 - 1 / 2) / 0.1)) +
    U() / 2 * (tanh((y - 1 + 1 / 2) / 0.1) - tanh((y - 1 - 1 / 2) / 0.1))

u_C(y) =
    U() / 2 * (tanh(((y + 1.0) / 1 + 1 / 2) / 0.1) - tanh(((y + 1.0) / 1 - 1 / 2) / 0.1)) +
    U() / 4 * (tanh(((y - 1.5) / 2 + 1 / 2) / 0.2) - tanh(((y - 1.5) / 2 - 1 / 2) / 0.2))

u_D(y) =
    U() / 2 * (tanh(((y + 1.0) / 1 + 1 / 2) / 0.1) - tanh(((y + 1.0) / 1 - 1 / 2) / 0.1)) -
    U() / 4 * (tanh(((y - 1.5) / 2 + 1 / 2) / 0.2) - tanh(((y - 1.5) / 2 - 1 / 2) / 0.2))

# u(y) = u_A(y)
# u(y) = u_B(y)
u(y) = u_C(y)
# u(y) = u_D(y)

# Random noise to stimulate turbulence
u(x, y) = (1 + 0.1 * (rand() - 1 / 2)) * u(y)

# # Boundary conditions: Unsteady BC requires time derivatives
# u_bc(x, y, t) = x ≈ 0.0 ? u(x, y) : 0.0
# v_bc(x, y, t) = 0.0
# bc_type = (;
#     u = (; x = (:periodic, :periodic), y = (:symmetric, :symmetric)),
#     v = (; x = (:periodic, :periodic), y = (:pressure, :pressure)),
# )

# A 2D grid is a Cartesian product of two vectors
n = 64
## n = 128
## n = 256
x = LinRange(0.0, 16.0, 4n)
y = LinRange(-10.0, 10.0, 5n)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model);
# setup = Setup(x, y; viscosity_model, u_bc, v_bc, bc_type);

# Since the grid is uniform and identical for x and y, we may use a specialized
# Fourier pressure solver
pressure_solver = FourierPressureSolver(setup)

# Time interval
t_start, t_end = tlims = (0.0, 1.0)

# Initial conditions
initial_velocity_u(x, y) = u(x, y)
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
    pressure_solver,
);
V, p = V₀, p₀

# Real time plot: Streamwise average and spectrum
o = StateObserver(1, V₀, p₀, t_start)
(; indu, yu, yin, Nux_in, Nuy_in) = setup.grid

umean = @lift begin
    V, p, t = $(o.state)
    u = V[indu]
    sleep(0.001)
    reshape(sum(reshape(u, size(yu)); dims = 1), :) ./ (Nux_in * U())
end

K = Nux_in ÷ 2
k = 1:(K - 1)

# Find energy spectrum where y = 0
n₀ = Nuy_in ÷ 2
E₀ = @lift begin
    V, p, t = $(o.state)
    u = V[indu]
    u_y = reshape(u, size(yu))[:, n₀]
    abs.(fft(u_y .^ 2))[k .+ 1]
end

# Find energy spectrum where y = 1
n₁ = argmin(n -> abs(yin[n] .- 1), 1:Nuy_in)
E₁ = @lift begin
    V, p, t = $(o.state)
    u = V[indu]
    u_y = reshape(u, size(yu))[:, n₁]
    abs.(fft(u_y .^ 2))[k .+ 1]
end

fig = Figure()
ax = Axis(
    fig[1, 1];
    title = "Mean streamwise flow",
    xlabel = "y",
    ylabel = L"\langle u \rangle / U_0",
)
lines!(ax, yu[1, :], umean)
ax = Axis(
    fig[1, 2];
    title = "Streamwise energy spectrum",
    xscale = log10,
    yscale = log10,
    xlabel = L"k_x",
    ylabel = L"\hat{U}_{cl} / U_0",
)
# ylims!(ax, (10^(0.0), 10^4.0))
ksub = k[10:end]
lines!(ax, ksub, 1000 .* ksub .^ (-3 / 5); label = L"k^{-3/5}")
lines!(ax, ksub, 1e7 .* ksub .^ -3; label = L"k^{-3}")
scatter!(ax, k, E₀; label = "y = $(yin[n₀])")
scatter!(ax, k, E₁; label = "y = $(yin[n₁])")
axislegend(ax; position = :lb)
fig

# Real time plot: Other option, just plot field
observer = StateObserver(1, V₀, p₀, t_start)
real_time_plot(observer, setup)

# Iteration processors
logger = Logger()
writer = VTKWriter(; nupdate = 1, dir = "output/$name", filename = "solution")
tracer = QuantityTracer()
## processors = [logger, observer, tracer, writer]
processors = [logger, observer, tracer]

# Solve unsteady problem
problem = UnsteadyProblem(setup, V, p, tlims);
V, p = solve(problem, RK44P2(); Δt = 0.001, processors, pressure_solver, inplace = true);
#md current_figure()

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

# Export to VTK
save_vtk(V, p, t_end, setup, "output/solution")

# Plot tracers
plot_tracers(tracer)

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, V₀, t_end)
plot_velocity(setup, V, t_end)

# Plot vorticity
plot_vorticity(setup, V, t_end)

# Plot stream function
plot_streamfunction(setup, V, t_end)
