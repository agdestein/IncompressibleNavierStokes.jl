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
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using LuxCUDA
using CUDA;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);

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
params =
    (; D = 2, Re, lims, nles = n, compression = 4, tburn, tsim, Δt = T(1e-4), ArrayType)
data_train = create_les_data(T; params..., nsim = ntrain);
data_valid = create_les_data(T; params..., nsim = nvalid);
data_test = create_les_data(T; params..., nsim = ntest);

length(data_train.u)
length(data_train.u[1])
length(data_train.u[1][1])
size(data_train.u[1][1][1])

o = Observable(data_train.u[1][end][1])
heatmap(o)
for i = 1:501
    o[] = data_train.u[1][i][1]
    # o[] = data_train.cF[1][i][1]
    sleep(0.001)
end

# # Save filtered DNS data
# jldsave("output/forced/data.jld2"; data_train, data_valid, data_test)

# # Load previous LES data
# data_train, data_valid, data_test = load("output/forced/data.jld2", "data_train", "data_valid", "data_test")

nt = length(data_train.u[1]) - 1

# Uniform periodic grid
pressure_solver = SpectralPressureSolver(setup);

closure, θ₀ = cnn(
    setup,

    # Radius
    [2, 2, 2, 2],

    # Channels
    [5, 5, 5, 2],

    # Activations
    [leakyrelu, leakyrelu, leakyrelu, identity],

    # Bias
    [true, true, true, false];
);

# closure, θ₀ = fno(
#     setup,
#
#     # Cut-off wavenumbers
#     [32, 32, 32, 32],
#
#     # Channel sizes
#     [24, 12, 8, 8],
#
#     # Fourier layer activations
#     [gelu, gelu, gelu, identity],
#
#     # Dense activation
#     gelu,
# );

closure.NN

# Create input/output arrays
function create_io_arrays(data, setup)
    nsample = length(data.u)
    nt = length(data.u[1]) - 1
    D = setup.grid.dimension()
    T = eltype(data.u[1][1][1])
    (; N) = setup.grid
    u = zeros(T, (N .- 2)..., D, nt + 1, nsample)
    c = zeros(T, (N .- 2)..., D, nt + 1, nsample)
    ifield = ntuple(Returns(:), D)
    for i = 1:nsample, j = 1:nt+1, α = 1:D
        copyto!(view(u, ifield..., α, j, i), view(data.u[i][j][α], setup.grid.Iu[α]))
        copyto!(view(c, ifield..., α, j, i), view(data.cF[i][j][α], setup.grid.Iu[α]))
    end
    reshape(u, (N .- 2)..., D, :), reshape(c, (N .- 2)..., D, :)
end

io_train = create_io_arrays(data_train, setup)
io_valid = create_io_arrays(data_valid, setup)
io_test = create_io_arrays(data_test, setup)

# Prepare training
θ = 1.0f-1 * cu(θ₀)
# θ = cu(θ₀)
opt = Optimisers.setup(Adam(1.0f-3), θ)
callbackstate = Point2f[]
randloss = create_randloss(mean_squared_error, closure, io_train...; nuse = 50, device = cu)

# Warm-up
randloss(θ)
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
    niter = 2000,
    ncallback = 10,
    callbackstate,
    callback = create_callback(closure, cu(io_valid)...; state = callbackstate),
);
GC.gc()
CUDA.reclaim()

Array(θ)

# # Save trained parameters
# jldsave("output/forced/theta_cnn.jld2"; theta = Array(θ))
# jldsave("output/forced/theta_fno.jld2"; theta = Array(θ))

# # Load trained parameters
# θθ = load("output/theta_cnn.jld2")
# θθ = load("output/theta_fno.jld2")
# copyto!(θ, θθ["theta"])

relative_error(closure(device(data_train.V[:, 1, :]), θ), device(data_train.cF[:, 1, :]))
relative_error(
    closure(device(data_train.V[:, end, :]), θ),
    device(data_train.cF[:, end, :]),
)
relative_error(closure(u_test, θ), c_test)

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

isample = 1
forcedsetup = (; setup..., force = data_train.force[:, isample]);

devsetup = device(forcedsetup);
V_nm, p_nm, outputs_nm = solve_unsteady(
    forcedsetup,
    data_test.V[:, 1, isample],
    data_test.p[:, 1, isample],
    (T(0), tsim);
    Δt = T(2e-4),
    processors = (
        field_plotter(devsetup; type = heatmap, nupdate = 1),
        energy_history_writer(forcedsetup),
        step_logger(; nupdate = 10),
    ),
    pressure_solver,
    inplace = false,
    device,
    devsetup,
)
ehist_nm = outputs_nm[2]

setup_fno = (; forcedsetup..., closure_model = V -> closure(V, θ))
devsetup = device(setup_fno);
V_fno, p_fno, outputs_fno = solve_unsteady(
    setup_fno,
    data_test.V[:, 1, isample],
    data_test.p[:, 1, isample],
    (T(0), tsim);
    Δt = T(2e-4),
    processors = (
        field_plotter(devsetup; type = heatmap, nupdate = 1),
        energy_history_writer(forcedsetup),
        step_logger(; nupdate = 10),
    ),
    pressure_solver,
    inplace = false,
    device,
    devsetup,
)
ehist_fno = outputs_fno[2]

state = Observable((; V = data_train.V[:, 1, 1], p = data_train.p[:, 1, 1], t = T(0)))
ehist = energy_history(forcedsetup, state)
for i = 2:nt+1
    t = (i - 1) / T(nt - 1) * tsim
    V = data_test.V[:, i, isample]
    p = data_test.p[:, i, isample]
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

V = data_train.V[:, end, isample]
p = data_train.p[:, end, isample]

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
