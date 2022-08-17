# # Decaying Homogeneous Isotropic Turbulence 3D (DHIT)
#
# This test case considers decaying homogeneous isotropic turbulence.

if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DifferentiableNavierStokes.jl")
    using .DifferentiableNavierStokes
end

using Revise
using Hardanger
using GLMakie
colorscheme!("GruvboxDark")
set_theme!(makie(gruvbox()))

using FFTW
using DifferentiableNavierStokes
using GLMakie
using LaTeXStrings
using Zygote

# Case name for saving results
name = "DecayingTurbulence3D"

# Floating point type for simulations
T = Float64

## Grid
N = 50
x = stretched_grid(0, 1, N)
y = stretched_grid(0, 1, N)
z = stretched_grid(0, 1, N)
grid = create_grid(x, y, z; T);

plot_grid(grid)

## Build setup and assemble operators
operators = build_operators(grid);

## Forcing parameters
bodyforce_u(x, y, z) = 0.0
bodyforce_v(x, y, z) = 0.0
bodyforce_w(x, y, z) = 0.0
force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v, bodyforce_w)

# Body force
DifferentiableNavierStokes.build_force!(force, grid)

## Viscosity model
viscosity_model = LaminarModel{T}(; Re = 2000)
# viscosity_model = MixingLengthModel{T}(; Re = 1000)
# viscosity_model = SmagorinskyModel{T}(; Re = 1000)
# viscosity_model = QRModel{T}(; Re = 1000)

setup = Setup(; grid, operators, viscosity_model, force);

## Pressure solver
# pressure_solver = DirectPressureSolver(setup)
# pressure_solver = CGPressureSolver(setup; maxiter = 500, abstol = 1e-8)
pressure_solver = FourierPressureSolver(setup)

## Initial conditions
K = N ÷ 2
σ = 30
# σ = 10
s = 5
function create_spectrum(K)
    a =
        1e6 * [
            1 / sqrt((2π)^3 * 3σ^2) *
            exp(-((i - s)^2 + (j - s)^2 + (k - s)^2) / 2σ^2) *
            exp(-2π * im * rand()) for i = 1:K, j = 1:K, k = 1:K
        ]
    [
        a reverse(a; dims = 2)
        reverse(a; dims = 1) reverse(a; dims = (1, 2));;;
        reverse(a; dims = 3) reverse(a; dims = (2, 3))
        reverse(a; dims = (1, 3)) reverse(a)
    ]
end
u = real.(ifft(create_spectrum(K)))
v = real.(ifft(create_spectrum(K)))
w = real.(ifft(create_spectrum(K)))
V = [reshape(u, :); reshape(v, :); reshape(w, :)]
f = setup.operators.M * V
p = zero(f)
Δp = DifferentiableNavierStokes.pressure_poisson(pressure_solver, f)
V .= V .- setup.grid.Ω⁻¹ .* (setup.operators.G * Δp)
(; Ω⁻¹) = grid
(; M) = setup.operators
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
plotter = RealTimePlotter(; nupdate = 100nupdate, fieldname = :vorticity, type = contour)
writer = VTKWriter(; nupdate = 100nupdate, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate)
processors = [logger, plotter, writer, tracer]
# processors = [logger, plotter, tracer]
# processors = [logger]

## Time interval
t_start, t_end = tlims = (0.0, 0.500)

## Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
# problem = UnsteadyProblem(setup, V, p, tlims);
V, p, t =
    @time solve(problem, RK44(); Δt = 0.0001, processors, inplace = false, pressure_solver)

function S(Re, V₀, p₀, tlims)
    viscosity_model = LaminarModel(Re)
    setup = Setup(; grid, operators, viscosity_model, force)
    problem = UnsteadyProblem(setup, V₀, p₀, tlims)
    V, p, t = solve(problem, RK44(); Δt = 0.0001, inplace = false, pressure_solver)
    V
end

function create_loss(V₀, p₀, V, tlims)
    loss(Re) = sum(abs2, S(Re, V₀, p₀, tlims) - V) / sum(abs2, V)
end

V = S(1.0, V₀, p₀, tlims)
loss = create_loss(V₀, p₀, V, tlims)

loss(2.0)
first(Zygote.gradient(loss, 2.0))

Re = 2.0
for i = 1:100
    Re = Re - 0.1 * first(Zygote.gradient(loss, Re))
    l = loss(Re)
    println("Iteration $i\tloss: $l\tRe: $Re")
end

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

