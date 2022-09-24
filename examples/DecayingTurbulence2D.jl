# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Decaying Homogeneous Isotropic Turbulence - 2D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

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
using LinearAlgebra

# Case name for saving results
name = "DecayingTurbulence2D"

# Viscosity model
viscosity_model = LaminarModel(; Re = 1e4)

# A 2D grid is a Cartesian product of two vectors
n = 200
x = LinRange(0.0, 1.0, n + 1)
y = LinRange(0.0, 1.0, n + 1)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model);

# Since the grid is uniform and identical for x and y, we may use a specialized
# Fourier pressure solver
pressure_solver = FourierPressureSolver(setup)

# Initial conditions
K = n ÷ 2
A = 1e6
σ = 30
## σ = 10
s = 5
function create_spectrum(K)
    a =
        A * [
            1 / sqrt((2π)^2 * 2σ^2) *
            exp(-((i - s)^2 + (j - s)^2) / 2σ^2) *
            exp(-2π * im * rand()) for i = 1:K, j = 1:K
        ]
    [
        a reverse(a; dims = 2)
        reverse(a; dims = 1) reverse(a)
    ]
end
u = real.(ifft(create_spectrum(K)))
v = real.(ifft(create_spectrum(K)))
V = [reshape(u, :); reshape(v, :)]
f = setup.operators.M * V
p = zero(f)

# Boundary conditions
bc_vectors = get_bc_vectors(setup, 0.0)
(; yM) = bc_vectors

# Make velocity field divergence free
(; Ω⁻¹) = setup.grid
(; G, M) = setup.operators
f = M * V + yM
Δp = pressure_poisson(pressure_solver, f)
V .-= Ω⁻¹ .* (G * Δp)
p = pressure_additional_solve(pressure_solver, V, p, 0.0, setup; bc_vectors)

V₀, p₀ = V, p

# Time interval
t_start, t_end = tlims = (0.0, 1.0)

# Iteration processors
logger = Logger()
observer = StateObserver(1, V₀, p₀, t_start)
writer = VTKWriter(; nupdate = 10, dir = "output/$name", filename = "solution")
tracer = QuantityTracer()
## processors = [logger, observer, tracer, writer]
processors = [logger, observer, tracer]

# Real time plot
rtp = real_time_plot(observer, setup)

# Plot energy history
(; Ωp) = setup.grid
_points = Point2f[]
points = @lift begin
    V, p, t = $(observer.state)
    up, vp = get_velocity(V, t, setup)
    up = reshape(up, :)
    vp = reshape(vp, :)
    E = up' * Diagonal(Ωp) * up + vp' * Diagonal(Ωp) * vp
    push!(_points, Point2f(t, E))
end
ehist = lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))

# Plot energy spectrum
k = 1:(K - 1)
kk = reshape([sqrt(kx^2 + ky^2) for kx ∈ k, ky ∈ k], :)
ehat = @lift begin
    V, p, t = $(observer.state)
    up, vp = get_velocity(V, t, setup)
    e = up .^ 2 .+ vp .^ 2
    reshape(abs.(fft(e)[k .+ 1, k .+ 1]), :)
end
espec = Figure()
ax =
    Axis(espec[1, 1]; xlabel = L"k", ylabel = L"\hat{e}(k)", xscale = log10, yscale = log10)
## ylims!(ax, (1e-20, 1))
scatter!(ax, kk, ehat; label = "Kinetic energy")
krange = LinRange(extrema(kk)..., 100)
lines!(ax, krange, 1e7 * krange .^ (-3); label = L"k^{-3}", color = :red)
axislegend(ax)
espec

# Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p = solve(problem, RK44(); Δt = 0.001, processors, pressure_solver, inplace = true);

# Real time plot
rtp

# Energy history
ehist

# Energy spectrum
espec

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
plot_velocity(setup, V, t_end)

# Plot vorticity
plot_vorticity(setup, V, t_end)
