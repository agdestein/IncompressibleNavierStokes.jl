# # Decaying Homogeneous Isotropic Turbulence 2D (DHIT)
#
# This test case considers decaying homogeneous isotropic turbulence.

if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

using FFTW
using IncompressibleNavierStokes
using LaTeXStrings

if haskey(ENV, "GITHUB_ACTIONS")
    using CairoMakie
else
    using GLMakie
end

# Case name for saving results
name = "DecayingTurbulence2D"

# Floating point type for simulations
T = Float64

# Viscosity model
viscosity_model = LaminarModel{T}(; Re = 10000)
## viscosity_model = MixingLengthModel{T}(; Re = 1000)
## viscosity_model = SmagorinskyModel{T}(; Re = 1000)
## viscosity_model = QRModel{T}(; Re = 1000)

# Convection model
convection_model = NoRegConvectionModel()
## convection_model = C2ConvectionModel()
## convection_model = C4ConvectionModel()
## convection_model = LerayConvectionModel()

# Boundary conditions
u_bc(x, y, t) = zero(x)
v_bc(x, y, t) = zero(x)
boundary_conditions = BoundaryConditions(
    u_bc,
    v_bc;
    bc_unsteady = false,
    bc_type = (;
        u = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        v = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
    ),
    T,
)

# Grid
N = 200
x = stretched_grid(0, 1, N)
y = stretched_grid(0, 1, N)
grid = Grid(x, y; boundary_conditions, T);

plot_grid(grid)

# Forcing parameters
bodyforce_u(x, y) = 0
bodyforce_v(x, y) = 0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

# Build setup and assemble operators
setup = Setup(; viscosity_model, convection_model, grid, force, boundary_conditions)

# Pressure solver
## pressure_solver = DirectPressureSolver(setup)
## pressure_solver = CGPressureSolver(setup; maxiter = 500, abstol = 1e-8)
pressure_solver = FourierPressureSolver(setup)

# Initial conditions
K = N ÷ 2
σ = 30
## σ = 10
s = 5
function create_spectrum(K)
    a =
        1e6 * [
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
Δp = IncompressibleNavierStokes.pressure_poisson(pressure_solver, f)
V .-= Ω⁻¹ .* (G * Δp)
p = IncompressibleNavierStokes.pressure_additional_solve(
    pressure_solver,
    V,
    p,
    0.0,
    setup;
    bc_vectors,
)

V₀, p₀ = V, p

# Iteration processors
nupdate = 1
logger = Logger()
plotter = RealTimePlotter(; nupdate, fieldname = :vorticity, type = heatmap)
writer = VTKWriter(; nupdate = 10nupdate, dir = "output/$name", filename = "solution")
tracer = QuantityTracer(; nupdate)
## processors = [logger, plotter, writer, tracer]
processors = [logger, plotter, tracer]

# Time interval
t_start, t_end = tlims = (0.0, 1.0)

# Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims);
V, p, = solve(problem, RK44(); Δt = 0.001, processors, inplace = true, pressure_solver)

# Kinetic energy spectrum
k = 1:K
u = reshape(V[grid.indu], N, N)
v = reshape(V[grid.indv], N, N)
e = u .^ 2 .+ v .^ 2
ehat = fft(e)[k, k]
kk = sqrt.(k .^ 2 .+ (k') .^ 2)

# Plot kinetic energy spectrum
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"k", ylabel = L"\hat{e}(k)", xscale = log10, yscale = log10)
## ylims!(ax, (1e-20, 1))
scatter!(ax, kk[:], abs.(ehat[:]); label = "Kinetic energy")
krange = LinRange(extrema(kk)..., 100)
lines!(ax, krange, 1e6 * krange .^ (-5 / 3); label = L"k^{-5/3}")
lines!(ax, krange, 1e7 * krange .^ (-3); label = L"k^{-3}")
axislegend(ax)
fig

# Post-process
plot_tracers(tracer)

#-

plot_pressure(setup, p)

#-

plot_velocity(setup, V, t_end)

#-

plot_vorticity(setup, V, tlims[2])

#-

## plot_streamfunction(setup, V, tlims[2])
