# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Train closure model
#
# Here, we consider a periodic box ``[0, 1]^2``. It is discretized with a
# uniform Cartesian grid with square cells.

using GLMakie
using IncompressibleNavierStokes
using JLD2
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Random
using Zygote

# Floating point precision
T = Float32

# # To use CPU: Do not move any arrays
# device = identity

# To use GPU, use `cu` to move arrays to the GPU.
# Note: `cu` converts to Float32
using CUDA
using LuxCUDA
device = cu

# Setup
n = 128
lims = T(0), T(1)
viscosity_model = LaminarModel(; Re = T(2_000))
t_burn = T(0.1)
t_sim = T(0.1)

# Build LES setup and assemble operators
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
setup = Setup(x, y; viscosity_model)

# Create LES data from DNS
filtered = create_les_data(
    T;
    viscosity_model,
    lims,
    n_les = n,
    compression = 4,
    n_ic = 10,
    t_burn,
    t_sim,
    Δt = T(2e-4),
    device,
)
jldsave("output/filtered/filtered.jld2"; filtered)

# Load previous LES data
filtered = load("output/filtered/filtered.jld2", "filtered")

size(filtered.V)

# Inspect data
plot_vorticity(setup, filtered.V[:, end, 1], T(0))

# Uniform periodic grid
pressure_solver_les = SpectralPressureSolver(setup)

# Compute commutator errors
_, n_t, n_ic = size(filtered.V)
bc_vectors = get_bc_vectors(setup, T(0))
commutator_error = zero(filtered.F)
pbar = filtered.p[:, 1, 1]
for i_t = 1:n_t, i_ic = 1:n_ic
    @info "Computing commutator error for time $i_t of $n_t, IC $i_ic of $n_ic"
    V = filtered.V[:, i_t, i_ic]
    F = filtered.F[:, i_t, i_ic]
    Fbar, = momentum(V, V, pbar, T(0), setup; bc_vectors, nopressure = true)
    commutator_error[:, i_t, i_ic] .= F .- Fbar
end

norm(commutator_error[:, 1, 1]) / norm(filtered.F[:, 1, 1])

# closure, θ₀ = cnn(
#     setup,
#     [5, 5, 5],
#     [2, 8, 8, 2],
#     [leakyrelu, leakyrelu, identity],
#     [true, true, false];
# )

closure, θ₀ = fno(
    # Setup
    setup,

    # Cut-off wavenumbers
    [8, 8, 8],

    # Channel sizes
    [16, 8, 8],

    # Fourier activations
    [gelu, gelu, identity],

    # Dense activation
    gelu,
)

@info "Closure model has $(length(θ₀)) parameters"

# Test data
V_test = device(reshape(filtered.V[:, 1:20, 1:2], :, 40))
c_test = device(reshape(commutator_error[:, 1:20, 1:2], :, 40))

# Prepare training
θ = 5.0f-2 * device(θ₀)
opt = Optimisers.setup(Adam(1.0f-2), θ)
callbackstate = Point2f[]
randloss = create_randloss(
    mean_squared_error,
    closure,
    filtered.V,
    commutator_error;
    nuse = 50,
    device,
)

# Warm-up
randloss(θ);
@time randloss(θ);
first(gradient(randloss, θ));
@time first(gradient(randloss, θ));

# Training
# Note: The states `opt`, `θ`, and `callbackstate`
# will not be overwritten until training is finished.
# This allows for cancelling with "Control-C" should errors explode.
(; opt, θ, callbackstate) = train(
    randloss,
    opt,
    θ;
    niter = 500,
    ncallback = 10,
    callbackstate,
    callback = create_callback(closure, V_test, c_test; state = callbackstate),
)

# jldsave("output/theta.jld2"; θ = Array(θ))
# θθ = load("output/theta.jld2")
# θθ = θθ["θ"]
# θθ = cu(θθ)
# θ .= θθ

relative_error(closure(V_test, θ), c_test)

devsetup = device(setup);

V_nm, p_nm, outputs_nm = solve_unsteady(
    setup,
    filtered.V[:, 1, 1],
    filtered.p[:, 1, 1],
    (T(0), t_sim);
    Δt = T(2e-4),
    processors = (
        step_logger(; nupdate = 10),
        field_plotter(devsetup; type = heatmap, nupdate = 1),
    ),
    pressure_solver = pressure_solver_les,
    inplace = false,
    device,
    devsetup,
)

V_fno, p_fno, outputs_fno = solve_unsteady(
    (; setup..., closure_model = V -> closure(V, θ)),
    filtered.V[:, 1, 1],
    filtered.p[:, 1, 1],
    (T(0), t_sim);
    Δt = T(2e-4),
    processors = (
        step_logger(; nupdate = 10),
        field_plotter(devsetup; type = heatmap, nupdate = 1),
    ),
    pressure_solver = pressure_solver_les,
    inplace = false,
    device,
    devsetup,
)

V = filtered.V[:, end, 1]
p = filtered.p[:, end, 1]

relative_error(V_nm, V)
relative_error(V_fno, V)

plot_vorticity(setup, V_nm, t_sim)
plot_vorticity(setup, V_fno, t_sim)
plot_vorticity(setup, V, t_sim)

CUDA.memory_status()
GC.gc()
CUDA.reclaim()
