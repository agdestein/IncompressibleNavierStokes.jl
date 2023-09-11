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
Re = T(6_000)
tburn = T(0.05)
tsim = T(0.05)

# Build LES setup and assemble operators
x = LinRange(lims..., n + 1)
y = LinRange(lims..., n + 1)
setup = Setup(x, y; Re);

# Number of simulations
ntrain = 10
nvalid = 2
ntest = 5

# Create LES data from DNS
params = (;
    Re,
    lims,
    nles = n,
    compression = 4,
    tburn,
    tsim,
    Δt = T(1e-4),
    device,
)
data_train = create_les_data(T; params..., nsim = ntrain)
data_valid = create_les_data(T; params..., nsim = nvalid)
data_test = create_les_data(T; params..., nsim = ntest)

# jldsave("output/filtered/data.jld2"; data_train, data_valid, data_test)

# Load previous LES data
data_train, data_valid, data_test = load("output/filtered/data.jld2", "data_train", "data_valid", "data_test")

nt = size(data_train.V, 2) - 1

size(data_train.V)
size(data_valid.V)
size(data_test.V)

# Inspect data
plot_vorticity(setup, data_valid.V[:, 1, 1], T(0))
plot_vorticity(setup, data_valid.V[:, end, 1], T(0))
norm(data_valid.cF[:, 1, 1]) / norm(data_valid.F[:, 1, 1])
norm(data_valid.cF[:, end, 1]) / norm(data_valid.F[:, end, 1])

# Uniform periodic grid
pressure_solver = SpectralPressureSolver(setup);

q = data_valid.V
# q = data_train.FG
q = q[:, :, 1]
q = selectdim(reshape(q, n, n, 2, :), 3, 1)

qc = reshape(selectdim(data_valid.cF, 3, 2), n, n, :)

obs = Observable(randn(T, 1, 1))
# obs = Observable(selectdim(q, 3, 1))
# obs = Observable([selectdim(q, 3, 1) selectdim(qc, 3, 1)])
fig = heatmap(obs)
fig

# for snap in eachslice(q; dims = 3)
for (s1, s2) in zip(eachslice(q; dims = 3), eachslice(qc; dims = 3))
    obs[] = s1
    # obs[] = [s1 s2]
    autolimits!(fig.axis)
    sleep(0.005)
end

heatmap(selectdim(reshape(data_valid.force[:, 1], n, n, 2), 3, 1))

fx, fy = eachslice(reshape(data_valid.force[:, 1], n, n, 2); dims = 3)
heatmap(fx)
arrows(x, y, fx, fy; lengthscale = 1.0f0)

# closure, θ₀ = cnn(
#     setup,
#
#     # Radius
#     [2, 2, 2, 2],
#
#     # Channels
#     [64, 64, 64, 2],
#
#     # Activations
#     [leakyrelu, leakyrelu, leakyrelu, identity],
#
#     # Bias
#     [true, true, true, false];
# )

closure, θ₀ = fno(
    setup,

    # Cut-off wavenumbers
    [32, 32, 32, 32],

    # Channel sizes
    [24, 12, 8, 8],

    # Fourier layer activations
    [gelu, gelu, gelu, identity],

    # Dense activation
    gelu,
);

@info "Closure model has $(length(θ₀)) parameters"

# Test data
V_test = device(reshape(data_test.V[:, 1:20, 1:2], :, 40))
c_test = device(reshape(data_test.cF[:, 1:20, 1:2], :, 40))

# Prepare training
θ = 1.0f-1 * device(θ₀)
# θ = device(θ₀)
opt = Optimisers.setup(Adam(1.0f-3), θ)
callbackstate = Point2f[]
randloss = create_randloss(
    mean_squared_error,
    closure,
    data_train.V,
    data_train.cF;
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
    niter = 1000,
    ncallback = 10,
    callbackstate,
    callback = create_callback(closure, V_test, c_test; state = callbackstate),
)
GC.gc()
CUDA.reclaim()

Array(θ)

# # Save trained parameters
# jldsave("output/theta.jld2"; θ = Array(θ))

# # Load trained parameters
# θθ = load("output/theta.jld2")
# θθ = θθ["θ"]
# θθ = cu(θθ)
# θ .= θθ

relative_error(closure(device(data_train.V[:, 1, :]), θ), device(data_train.cF[:, 1, :]))
relative_error(closure(device(data_train.V[:, end, :]), θ), device(data_train.cF[:, end, :]))
relative_error(closure(V_test, θ), c_test)

function energy_history(setup, state)
    (; Ωp) = setup.grid
    points = Point2f[]
    on(state) do (; V, p, t)
        V = Array(V)
        vels = get_velocity(setup, V, t)
        vels = reshape.(vels, :)
        E = sum(vel -> sum(@. Ωp * vel^2), vels)
        push!(points, Point2f(t, E))
    end
    points
end

energy_history_writer(setup; nupdate = 1, kwargs...) =
    processor(state -> energy_history(setup, state; kwargs...); nupdate)

devsetup = device(setup);
V_nm, p_nm, outputs_nm = solve_unsteady(
    setup,
    data_train.V[:, 1, 1],
    data_train.p[:, 1, 1],
    (T(0), tsim);
    Δt = T(2e-4),
    processors = (
        field_plotter(devsetup; type = heatmap, nupdate = 1),
        energy_history_writer(setup),
        step_logger(; nupdate = 10),
    ),
    pressure_solver,
    inplace = false,
    device,
    devsetup,
)
ehist_nm = outputs_nm[2]

setup_fno = (; setup..., closure_model = V -> closure(V, θ))
devsetup = device(setup_fno);
V_fno, p_fno, outputs_fno = solve_unsteady(
    setup_fno,
    data_train.V[:, 1, 1],
    data_train.p[:, 1, 1],
    (T(0), tsim);
    Δt = T(2e-4),
    processors = (
        field_plotter(devsetup; type = heatmap, nupdate = 1),
        energy_history_writer(setup),
        step_logger(; nupdate = 10),
    ),
    pressure_solver,
    inplace = false,
    device,
    devsetup,
)
ehist_fno = outputs_fno[2]

state = Observable((; V = data_train.V[:, 1, 1], p = data_train.p[:, 1, 1], t = T(0)))
ehist = energy_history(setup, state)
for i = 2:nt+1
    t = (i - 1) / T(nt - 1) * tsim
    V = data_train.V[:, i, 1]
    p = data_train.p[:, i, 1]
    state[] = (; V, p, t)
end
ehist

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "Kinetic energy")
lines!(ax, ehist; label = "Reference")
lines!(ax, ehist_nm; label = "No closure")
lines!(ax, ehist_fno; label = "FNO")
axislegend(ax)
fig

save("output/train/energy.png", fig)

V = data_train.V[:, end, 1]
p = data_train.p[:, end, 1]

relative_error(V_nm, V)
relative_error(V_fno, V)

box = [
    Point2f(0.72, 0.42),
    Point2f(0.81, 0.42),
    Point2f(0.81, 0.51),
    Point2f(0.72, 0.51),
    Point2f(0.72, 0.42),
]

plot_vorticity(setup, V, tsim)
lines!(box; color = :red)
current_figure()

save("output/train/vorticity.png", current_figure())

plot_vorticity(setup, V_nm, tsim)
lines!(box; color = :red)
current_figure()

save("output/train/vorticity_nm.png", current_figure())

plot_vorticity(setup, V_fno, tsim)
lines!(box; color = :red)
current_figure()

save("output/train/vorticity_fno.png", current_figure())

heatmap(vcat(
    selectdim(reshape(V_nm, n, n, 2), 3, 1),
    # selectdim(reshape(V_fno, n, n, 2), 3, 1),
    selectdim(reshape(V, n, n, 2), 3, 1),
))

CUDA.memory_status()
GC.gc()
CUDA.reclaim()
