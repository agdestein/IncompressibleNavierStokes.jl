# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

using GLMakie
using IncompressibleNavierStokes
using LinearAlgebra

# Floating point precision
T = Float32

# To use CPU: Do not move any arrays
device = identity

# # To use GPU, use `cu` to move arrays to the GPU.
# # Note: `cu` converts to Float32
# using CUDA
# device = cu

# Viscosity model
Re = T(2_000)
laminar = LaminarModel(; Re)
smagorinsky = SmagorinskyModel(; Re, C_s = T(0.173))

# A 2D grid is a Cartesian product of two vectors
s = 2
n_coarse = 256
n = s * n_coarse

lims = T(0), T(1)
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
plot_grid(x, y)

x_coarse = x[1:s:end]
y_coarse = y[1:s:end]
plot_grid(x_coarse, y_coarse)

# Build setup and assemble operators
setup = Setup(x, y; viscosity_model = laminar);
devsetup = device(setup);
setup_coarse = Setup(x_coarse, y_coarse; viscosity_model = laminar);

# Filter
(; KV, Kp) = operator_filter(setup.grid, setup.boundary_conditions, s);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
pressure_solver = SpectralPressureSolver(setup);
pressure_solver_coarse = SpectralPressureSolver(setup_coarse);

# Initial conditions
V₀, p₀ = random_field(setup; A = T(10_000_000), σ = T(30), s = 5, pressure_solver);

filter_saver(setup, KV, Kp; nupdate = 1, bc_vectors = get_bc_vectors(setup, T(0))) =
    processor(
        function (step_observer)
            T = eltype(setup.grid.x)
            _V = fill(zeros(T, 0), 0)
            _F = fill(zeros(T, 0), 0)
            _FG = fill(zeros(T, 0), 0)
            _p = fill(zeros(T, 0), 0)
            _t = fill(zero(T), 0)
            @lift begin
                (; V, p, t) = $step_observer
                F, = momentum(V, V, p, t, setup; bc_vectors, nopressure = true)
                FG, = momentum(V, V, p, t, setup; bc_vectors, nopressure = false)
                push!(_V, KV * Array(V))
                push!(_F, KV * Array(F))
                push!(_FG, KV * Array(FG))
                push!(_p, Kp * Array(p))
                push!(_t, t)
            end
            (; V = _V, F = _F, FG = _FG, p = _p, t = _t)
        end;
        nupdate,
    )

# Iteration processors
processors = (
    filter_saver(devsetup, KV, Kp; bc_vectors = device(get_bc_vectors(setup, T(0)))),
    field_plotter(devsetup; type = image, nupdate = 1),
    # energy_history_plotter(devsetup; nupdate = 20, displayfig = false),
    # energy_spectrum_plotter(devsetup; nupdate = 20, displayfig = false),
    # animator(devsetup; nupdate = 16),
    ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
    ## field_saver(setup; nupdate = 10),
    step_logger(; nupdate = 10),
);

processors_coarse = (
    field_plotter(device(setup_coarse); type = image, nupdate = 1),
    step_logger(; nupdate = 10),
);

# Time interval
t_start, t_end = tlims = T(0), T(0.1)

# Solve unsteady problem
V, p, outputs = solve_unsteady(
    setup,
    V₀,
    p₀,
    # V, p,
    tlims;
    Δt = T(2e-4),
    processors,
    pressure_solver,
    inplace = true,
    device,
);

# V₀, p₀ = V, p

Vbar = KV * V
pbar = Kp * p

Vbar_nomodel, pbar_nomodel, outputs_lam = solve_unsteady(
    setup_coarse,
    KV * V₀,
    Kp * p₀,
    tlims;
    Δt = T(2e-4),
    processors = processors_coarse,
    pressure_solver = pressure_solver_coarse,
    inplace = true,
    device,
);

Vbar_smag, pbar_smag, outputs_smag = solve_unsteady(
    (; setup_coarse..., viscosity_model = smagorinsky),
    KV * V₀,
    Kp * p₀,
    tlims;
    Δt = T(2e-4),
    processors = processors_coarse,
    pressure_solver = pressure_solver_coarse,
    inplace = true,
    device,
);

norm(Vbar_nomodel - Vbar) / norm(Vbar)
norm(Vbar_smag - Vbar) / norm(Vbar)

# Filtered quantities
filtered = outputs[1]
filtered.V[end]
