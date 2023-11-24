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
device = identity
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using LuxCUDA
using CUDA;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);
device = cu

# Parameters
params = (;
    D = 2,
    Re = T(6_000),
    lims = (T(0), T(1)),
    nles = 128,
    compression = 8,
    tburn = T(0.05),
    tsim = T(0.2),
    Δt = T(1e-4),
    ArrayType,
    # ic_params = (; A = T(20_000_000), σ = T(5.0), s = T(3)),
    ic_params = (; A = T(10_000_000)),
)

# Create LES data from DNS
data_train = create_les_data(T; params..., nsim = 10);
data_valid = create_les_data(T; params..., nsim = 1);
data_test = create_les_data(T; params..., nsim = 5);

# Inspect data
isim = 1
α = 1
# j = 13
o = Observable(data_train.u[isim][1][α])
# o = Observable(data_train.u[isim][1][α][:, :, j])
heatmap(o)
for i = 1:length(data_train.u[isim])
    o[] = data_train.u[isim][i][α]
    # o[] = data_train.u[isim][i][α][:, :, j]
    sleep(0.001)
end

# # Save filtered DNS data
# jldsave("output/forced/data.jld2"; data_train, data_valid, data_test)

# # Load previous LES data
# data_train, data_valid, data_test = load("output/forced/data.jld2", "data_train", "data_valid", "data_test")

# Build LES setup and assemble operators
x = ntuple(α -> LinRange(params.lims..., params.nles + 1), params.D)
setup = Setup(x...; params.Re, ArrayType);

# Uniform periodic grid
pressure_solver = SpectralPressureSolver(setup);

closure, θ₀ = cnn(;
    setup,
    radii = [2, 2, 2, 2],
    channels = [32, 32, 32, params.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, identity],
    bias = [true, true, true, false],
);
closure.NN

sample = io_train[1][:, :, :, 1:5]
closure(sample, θ₀) |> size

θ.layer_5
θ.layer_6

# closure, θ₀ = fno(;
#     setup,
#
#     # Cut-off wavenumbers
#     k = [8, 8, 8, 8],
#
#     # Channel sizes
#     c = [16, 16, 16, 16],
#
#     # Fourier layer activations
#     σ = [gelu, gelu, gelu, identity],
#
#     # Dense activation
#     ψ = gelu,
# );
# closure.NN

# Create input/output arrays
io_train = create_io_arrays(data_train, setup);
io_valid = create_io_arrays(data_valid, setup);
io_test = create_io_arrays(data_test, setup);

size(io_train[1])

# Prepare training
θ = T(1.0e-1) * device(θ₀);
# θ = device(θ₀);
opt = Optimisers.setup(Adam(T(1.0e-3)), θ);
callbackstate = Point2f[];
randloss = create_randloss(mean_squared_error, closure, io_train...; nuse = 50, device);

# Warm-up
randloss(θ)
@time randloss(θ);
first(gradient(randloss, θ));
@time first(gradient(randloss, θ));
GC.gc()
CUDA.reclaim()

# Training
# Note: The states `opt`, `θ`, and `callbackstate`
# will not be overwritten until training is finished.
# This allows for cancelling with "Control-C" should errors explode.
(; opt, θ, callbackstate) = train(
    randloss,
    opt,
    θ;
    niter = 1000,
    ncallback = 20,
    callbackstate,
    callback = create_callback(closure, device(io_valid)...; state = callbackstate),
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

relerr_track(uref, setup) = processor() do state
    (; dimension, x, Ip) = setup.grid
    D = dimension()
    T = eltype(x[1])
    e = Ref(T(0))
    on(state) do (; u, n)
        a, b = T(0), T(0)
        for α = 1:D
            # @show size(uref[n + 1])
            a += sum(abs2, u[α][Ip] - uref[n+1][α][Ip])
            b += sum(abs2, uref[n+1][α][Ip])
        end
        e[] += sqrt(a) / sqrt(b) / (length(uref) - 1)
    end
    e
end

u, u₀, p₀ = nothing, nothing, nothing
u = device.(data_test.u[1])
u₀ = device(data_test.u[1][1])
p₀ = pressure_additional_solve(pressure_solver, u₀, T(0), setup)
length(u)

u_nm, p_nm, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    (T(0), params.tsim);
    Δt = data_test.Δt,
    pressure_solver,
    processors = (
        relerr = relerr_track(u, setup),
        log = timelogger(; nupdate = 1),
    ),
)
relerr_nm = outputs.relerr[]

u_cnn, p_cnn, outputs = solve_unsteady(
    (; setup..., closure_model = create_neural_closure(closure, θ, setup)),
    u₀,
    p₀,
    (T(0), params.tsim);
    Δt = data_test.Δt,
    pressure_solver,
    processors = (
        relerr = relerr_track(u, setup),
        log = timelogger(; nupdate = 1),
    ),
)
relerr_cnn = outputs.relerr[]

relerr_nm
relerr_cnn

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
