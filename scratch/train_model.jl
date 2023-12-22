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

using Adapt
using GLMakie
using IncompressibleNavierStokes
using JLD2
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Random
using Zygote

# Random number generator
rng = Random.default_rng()
Random.seed!(rng, 123)

set_theme!(; GLMakie = (; scalefactor = 1.5))

# Floating point precision
T = Float64

# Array type
ArrayType = Array
device = identity
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using LuxCUDA
using CUDA;
T = Float32;
# T = Float64;
ArrayType = CuArray;
CUDA.allowscalar(false);
# device = cu
device = x -> adapt(CuArray{T}, x)

# Parameters
# nles = 50
# nles = 64
# nles = 128
# ndns = 200
nles = 256
params = (;
    D = 2,
    Re = T(6_000),
    lims = (T(0), T(1)),
    nles = [(nles, nles)],
    # ndns = (512, 512),
    ndns = (1024, 1024),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(1e-4),
    savefreq = 5,
    ArrayType,
    PSolver = SpectralPressureSolver,
)

# Create LES data from DNS
data_train = [create_les_data(T; params...) for _ = 1:3];
data_valid = [create_les_data(T; params...) for _ = 1:1];
data_test = [create_les_data(T; params..., tsim = T(0.2)) for _ = 1:1];

data_test[1].u[1][1][1]

# # Save filtered DNS data
# jldsave("output/forced/data.jld2"; data_train, data_valid, data_test)

# # Load previous LES data
# data_train, data_valid, data_test = load("output/forced/data.jld2", "data_train", "data_valid", "data_test")

# Build LES setup and assemble operators
x = ntuple(α -> LinRange(params.lims..., nles + 1), params.D)
setup = Setup(x...; params.Re, ArrayType);

# Uniform periodic grid
psolver = SpectralPressureSolver(setup);

# Inspect data
(; Ip) = setup.grid;
field = data_train[1].u[1];
α = 2
# j = 13
o = Observable(field[1][α][Ip])
# o = Observable(field[1][α][:, :, j])
heatmap(o)
for i = 1:length(field)
    o[] = field[i][α][Ip]
    # o[] = field[i][α][:, :, j]
    sleep(0.001)
end

# Inspect data
field = data_train[1].u[1];
u = device(field[1])
o = Observable((; u, t = nothing))
fieldplot(
    o;
    setup,
    # fieldname = :velocity,
    # fieldname = 2,
)
# energy_spectrum_plot(o; setup)
for i = 1:length(field)
    o[] = (; o[]..., u = device(field[i]))
    sleep(0.001)
end

# Create input/output arrays
io_train = create_io_arrays(data_train, [setup]);
io_valid = create_io_arrays(data_valid, [setup]);
io_test = create_io_arrays(data_test, [setup]);

size(io_train[1].u)
size(io_valid[1].u)
size(io_test[1].u)

Base.summarysize(io_train) / 1e9

closure, θ₀ = cnn(;
    setup,
    radii = [2, 2, 2, 2],
    channels = [5, 5, 5, params.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, false],
    rng,
);
closure.chain

sample = io_train[1].u[:, :, :, 1:5]
closure(sample, θ₀) |> size

θ₀.layer_5
θ₀.layer_6

# closure, θ₀ = fno(;
#     setup,
#     kmax = [20, 20, 20, 20],
#     c = [24, 12, 8, 8],
#     # σ = [gelu, gelu, gelu, identity],
#     σ = [leakyrelu, leakyrelu, leakyrelu, identity],
#     # σ = [tanh, tanh, tanh, identity],
#     # ψ = gelu,
#     ψ = tanh,
#     rng,
# );
#
# closure.chain
#
# θ₀.layer_4.spectral_weights |> size

# θ₀
# θ₀ = 2*θ₀
# θ₀.layer_6 ./= 20

# Prepare training
θ = T(1.0e-1) * device(θ₀);
# θ = device(θ₀);
# θ = 2 * device(θ₀);
opt = Optimisers.setup(Adam(T(1.0e-3)), θ);
callbackstate = Point2f[];
it = rand(1:size(io_valid[1].u, 4), 50);
validset = map(v -> v[:, :, :, it], io_valid[1]);

# A-priori loss
loss = createloss(mean_squared_error, closure);
dataloader = createdataloader(io_train[1]; batchsize = 50, device);
dataloaders = [dataloader]
dataloader()

# A-posteriori loss
loss = IncompressibleNavierStokes.create_trajectory_loss(; setup, psolver, closure);
dataloaders = [
    IncompressibleNavierStokes.createtrajectoryloader(data_train; device, nunroll = 20)
    for _ = 1:4
];
loss(dataloaders[1](), device(θ₀))

# Warm-up
loss(dataloaders[1](), θ)
@time loss(dataloaders[1](), θ);
b = dataloaders[1]();
first(gradient(θ -> loss(b, θ), θ));
@time first(gradient(θ -> loss(b, θ), θ));
GC.gc()
CUDA.reclaim()

map() do
    i = 3
    # h = 1000 * sqrt(eps(T))
    h = cbrt(eps(T))
    θ1 = copy(θ)
    θ2 = copy(θ)
    CUDA.@allowscalar θ1[i] -= h / 2
    CUDA.@allowscalar θ2[i] += h / 2
    b = dataloaders[1]()
    @show loss(b, θ2) loss(b, θ1)
    a = (loss(b, θ2) - loss(b, θ1)) / h
    b = CUDA.@allowscalar first(gradient(θ -> loss(b, θ), θ))[i]
    [a; b]
end

# Training
# Note: The states `opt`, `θ`, and `callbackstate`
# will not be overwritten until training is finished.
# This allows for cancelling with "Control-C" should errors explode.
(; opt, θ, callbackstate) = train(
    dataloaders,
    loss,
    opt,
    θ;
    niter = 1000,
    ncallback = 10,
    callbackstate,
    callback = create_callback(closure, device(validset)...; state = callbackstate),
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

function relerr(u, uref, setup)
    (; dimension, Ip) = setup.grid
    D = dimension()
    a, b = T(0), T(0)
    for α = 1:D
        a += sum(abs2, u[α][Ip] - uref[α][Ip])
        b += sum(abs2, uref[α][Ip])
    end
    sqrt(a) / sqrt(b)
end

u, u₀, = nothing, nothing
u = device.(data_test[1].u[1]);
u₀ = device(data_test[1].u[1][1]);
Δt = data_test[1].t[2] - data_test[1].t[1]
tlims = extrema(data_test[1].t)
length(u)
length(data_test[1].t)

state_nm, outputs = solve_unsteady(
    setup,
    u₀,
    tlims;
    Δt,
    psolver,
    processors = (;
        relerr = relerr_trajectory(u, setup),
        log = timelogger(; nupdate = 1000),
    ),
)
relerr_nm = outputs.relerr[]

state_cnn, outputs = solve_unsteady(
    (; setup..., closure_model = wrappedclosure(closure, θ, setup)),
    u₀,
    tlims;
    Δt,
    psolver,
    processors = (relerr = relerr_trajectory(u, setup), log = timelogger(; nupdate = 1)),
)
relerr_cnn = outputs.relerr[]

relerr_nm
relerr_cnn

# dnm = relerr_nm
# dcnn = relerr_cnn

dnm
dcnn

function energy_history(setup, state)
    (; Ωp) = setup.grid
    points = Point2f[]
    on(state) do (; V, t)
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
V_nm, outputs_nm = solve_unsteady(
    forcedsetup,
    data_test.V[:, 1, isample],
    (T(0), tsim);
    Δt = T(2e-4),
    processors = (
        field_plotter(devsetup; type = heatmap, nupdate = 1),
        energy_history_writer(forcedsetup),
        step_logger(; nupdate = 10),
    ),
    psolver,
    inplace = false,
    device,
    devsetup,
)
ehist_nm = outputs_nm[2]

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
