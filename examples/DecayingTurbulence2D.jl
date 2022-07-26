# # Decaying Homogeneous Isotropic Turbulence 2D (DHIT)
#
# This test case considers decaying homogeneous isotropic turbulence.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DifferentiableNavierStokes.jl")
    using .DifferentiableNavierStokes
end

using FFTW
using DifferentiableNavierStokes
using GLMakie
using LaTeXStrings
using Zygote

# Case name for saving results
name = "DecayingTurbulence2D"

# Floating point type for simulations
T = Float64

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 10000)
# viscosity_model = MixingLengthModel{T}(; Re = 1000)
# viscosity_model = SmagorinskyModel{T}(; Re = 1000)
# viscosity_model = QRModel{T}(; Re = 1000)

## Forcing parameters
bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

## Pressure solver
# pressure_solver = DirectPressureSolver{T}()
# pressure_solver = CGPressureSolver{T}(; maxiter = 500, abstol = 1e-8)
pressure_solver = FourierPressureSolver{T}()

## Grid
N = 500
x = stretched_grid(0, 1, N)
y = stretched_grid(0, 1, N)
grid = create_grid(x, y; T);

plot_grid(grid)

## Build setup and assemble operators
setup = Setup{T,2}(; viscosity_model, grid, force, pressure_solver);
build_operators!(setup);

## Initial conditions
u = zero(grid.xu)
v = zero(grid.xv)
p = zero(grid.xpp)
uhat = fft(u)
vhat = fft(v)
K = N ÷ 2
σ = 30
# σ = 10
s = 5
a =
    2e6 *
    # 2e5 *
    [
        1 / sqrt(4π)σ * exp(-((i - s)^2 + (j - s)^2) / 2σ^2) * exp(-2π * im * rand()) for
        i = 1:K, j = 1:K
    ]
uhat = [
    a reverse(a; dims = 2)
    reverse(a; dims = 1) reverse(a)
]
u = real.(ifft(uhat))
v = imag.(ifft(uhat))
V = [reshape(u, :); reshape(v, :)]
p = reshape(p, :)
f = setup.operators.M * V
Δp = DifferentiableNavierStokes.pressure_poisson(pressure_solver, f, setup)
V .-= setup.grid.Ω⁻¹ .* (setup.operators.G * Δp)
(; Ω⁻¹) = grid
(; M) = operators
# Momentum already contains G*p with the current p, we therefore
# effectively solve for the pressure difference
F = DifferentiableNavierStokes.momentum(V, p, 0.0, setup)
f = M * (Ω⁻¹ .* F)
Δp = DifferentiableNavierStokes.pressure_poisson(pressure_solver, f)
p = p + Δp

V₀, p₀ = V, p

## Iteration processors
nupdate = 1
logger = Logger()
plotter = RealTimePlotter(; nupdate, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 10nupdate, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate)
# processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]
# processors = [logger]

## Time interval
t_start, t_end = tlims = (0.0, 0.02)

## Solve unsteady problem
problem = UnsteadyProblem(setup, V, p, tlims);
V, p, t = @time solve(problem, RK44(); Δt = 0.0001, processors, inplace = false)

k = 1:K
u = reshape(V[grid.indu], N, N)
v = reshape(V[grid.indv], N, N)
e = u .^ 2 .+ v .^ 2
ehat = fft(e)[k, k]
kk = sqrt.(k .^ 2 .+ (k') .^ 2)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"k", ylabel = L"\hat{e}(k)", xscale = log10, yscale = log10)
# ylims!(ax, (1e-20, 1))
scatter!(ax, kk[:], abs.(ehat[:]); label = "Kinetic energy")
krange = LinRange(extrema(kk)..., 100)
lines!(ax, krange, 1e6 * krange .^ (-5 / 3); label = L"k^{-5/3}")
lines!(ax, krange, 1e7 * krange .^ (-3); label = L"k^{-3}")
axislegend(ax)
fig

## Post-process
plot_tracers(tracer)
plot_pressure(setup, p)
plot_velocity(setup, V, t_end)
plot_vorticity(setup, V, tlims[2])
plot_streamfunction(setup, V, tlims[2])

