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

using CairoMakie
using GLMakie
using IncompressibleNavierStokes
using JLD2
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Random
using Zygote
using SparseArrays
using KernelAbstractions
using FFTW

GLMakie.activate!()

set_theme!(; GLMakie = (; scalefactor = 1.5))

# Random number generator
rng = Random.default_rng()
Random.seed!(rng, 123)

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
get_params(nles) = (;
    D = 2,
    Re = T(6_000),
    lims = (T(0), T(1)),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(1e-4),
    nles,
    ndns = 2048,
    ArrayType,
)

params_train = (; get_params([64, 128, 256])..., savefreq = 5);
params_valid = (; get_params([128])..., savefreq = 10);
params_test = (; get_params([32, 64, 128, 256, 512, 1024])..., tsim = T(0.1), savefreq = 5);

# Create LES data from DNS
data_train = [create_les_data(T; params_train...) for _ = 1:5];
data_valid = [create_les_data(T; params_valid...) for _ = 1:1];
data_test = create_les_data(T; params_test...);

# # Save filtered DNS data
# jldsave("output/forced/data.jld2"; data_train, data_valid, data_test)

# # Load previous LES data
# data_train, data_valid, data_test = load("output/forced/data.jld2", "data_train", "data_valid", "data_test")

# Build LES setup and assemble operators
getsetups(params) = [
    Setup(
        ntuple(α -> LinRange(params.lims..., nles + 1), params.D)...;
        params.Re,
        params.ArrayType,
    ) for nles in params.nles
]
setups_train = getsetups(params_train);
setups_valid = getsetups(params_valid);
setups_test = getsetups(params_test);

# Create input/output arrays
io_train = create_io_arrays(data_train, setups_train);
io_valid = create_io_arrays(data_valid, setups_valid);

# Inspect data
ig = 5
# field, setup = data_train[1].u[ig], setups_train[ig];
# field, setup = data_valid[1].u[ig], setups_valid[ig];
field, setup = data_test.u[ig], setups_test[ig];
u = device(field[1]);
o = Observable((; u, p = similar(u[1], setup.grid.N), t = nothing));
fieldplot(
    o;
    setup,
    # fieldname = :velocity,
    # fieldname = 2,
)
# energy_spectrum_plot( o; setup,)
for i = 1:length(field)
    o[] = (; o[]..., u = device(field[i]))
    sleep(0.001)
end

GLMakie.activate!()
CairoMakie.activate!()

# Training data plot
fig = with_theme() do
    sample = data_train[1]
    fig = Figure()
    for (i, it) in enumerate((1, length(sample.t)))
        for (j, ig) in enumerate((1, 2, 3))
            setup = setups_train[ig]
            xf = Array.(getindex.(setup.grid.xp, setup.grid.Ip.indices))
            u = sample.u[ig][it] |> device
            ωp =
                IncompressibleNavierStokes.interpolate_ω_p(
                    IncompressibleNavierStokes.vorticity(u, setup),
                    setup,
                )[setup.grid.Ip] |> Array
            colorrange = IncompressibleNavierStokes.get_lims(ωp)
            opts = (;
                xticksvisible = false,
                xticklabelsvisible = false,
                yticklabelsvisible = false,
                yticksvisible = false,
            )
            i == 2 && (
                opts = (;
                    opts...,
                    xlabel = "x",
                    xticksvisible = true,
                    xticklabelsvisible = true,
                )
            )
            j == 1 && (
                opts = (;
                    opts...,
                    ylabel = "y",
                    yticklabelsvisible = true,
                    yticksvisible = true,
                )
            )
            ax = Axis(
                fig[i, j];
                opts...,
                title = "n = $(params_train.nles[ig]), t = $(round(sample.t[it]; digits = 1))",
                aspect = DataAspect(),
                limits = (params_test.lims..., params_test.lims...),
            )
            heatmap!(ax, xf..., ωp; colorrange)
        end
    end
    fig
end

save("training_data.pdf", fig)

# Plot training spectra
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    ig = 3
    setup = setups_train[ig]
    D = params_train.D
    backend = KernelAbstractions.get_backend(setup.grid.x[1])
    sample = data_train[1]
    K = size(setup.grid.Ip) .÷ 2
    kx = ntuple(α -> 1:K[α]-1, params_train.D)
    k = zero(similar(setup.grid.x[1], length.(kx)))
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)
    # Make averaging matrix
    naverage = 5^D
    i = sortperm(k)
    nbin, r = divrem(length(i), naverage)
    ib = KernelAbstractions.zeros(backend, Int, nbin * naverage)
    ia = KernelAbstractions.zeros(backend, Int, nbin * naverage)
    for j = 1:naverage
        copyto!(view(ia, (j-1)*nbin+1:j*nbin), collect(1:nbin))
        ib[(j-1)*nbin+1:j*nbin] = i[j:naverage:end-r]
    end
    vals = KernelAbstractions.ones(backend, T, nbin * naverage) / naverage
    A = sparse(ia, ib, vals, nbin, length(i))
    k = Array(A * k)
    # Build inertial slope above energy
    @show length(k)
    krange = collect(extrema(k[20:end]))
    # slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}") : (-T(5 / 3), L"$k^{-5/3}")
    slope, slopelabel = D == 2 ? (-T(3), "||k||₂⁻³") : (-T(5 / 3), L"k⁻⁵³")
    slopeconst = T(0)
    # Make plot
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xlabel = "||k||₂",
        ylabel = "e(k)",
        title = "Kinetic energy (n = $(params_train.nles[ig]))",
        xscale = log10,
        yscale = log10,
        xticks = [4, 8, 16, 32, 64, 128],
        limits = (extrema(k)..., T(1e-8), T(1)),
    )
    for (i, it) in enumerate((1, length(sample.t)))
        u = device.(sample.u[ig][it])
        up = IncompressibleNavierStokes.interpolate_u_p(u, setup)
        e = sum(up -> up[setup.grid.Ip] .^ 2, up)
        ehat = Array(
            A * reshape(abs.(fft(e)[ntuple(α -> kx[α] .+ 1, D)...]) ./ size(e, 1), :),
        )
        slopeconst = max(slopeconst, maximum(ehat ./ k .^ slope))
        lines!(ax, k, ehat; label = "t = $(round(sample.t[it]; digits = 1))")
    end
    inertia = 2 .* slopeconst .* krange .^ slope
    lines!(ax, krange, inertia; linestyle = :dash, label = slopelabel)
    axislegend(ax; position = :lb)
    autolimits!(ax)
    fig
end

save("training_spectra.pdf", fig)

closure, θ₀ = cnn(;
    setup = setups_train[1],
    radii = [2, 2, 2, 2, 2, 2],
    channels = [32, 32, 32, 32, 32, params_train.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, true, true, false],
);
closure.chain

# Prepare training
loss = createloss(mean_squared_error, closure);
dataloaders = createdataloader.(io_train; batchsize = 50, device);
# dataloaders[1]()
loss(dataloaders[1](), device(θ₀))
it = rand(1:size(io_valid[1].u, 4), 50);
validset = map(v -> v[:, :, :, it], io_valid[1]);

# Prepare training
θ = T(1.0e-1) * device(θ₀);
opt = Optimisers.setup(Adam(T(1.0e-3)), θ);
callbackstate = Point2f[];

# Training with multiple grids at the same time
(; opt, θ, callbackstate) = train(
    dataloaders,
    loss,
    opt,
    θ;
    niter = 1000,
    ncallback = 20,
    callbackstate,
    callback = create_callback(closure, device(validset)...; state = callbackstate),
);
GC.gc()
CUDA.reclaim()

# Extract parameters
θ_cnn_shared = θ;

# # Save trained parameters
# jldsave("output/multigrid/theta_cnn_shared.jld2"; theta = Array(θ_cnn_shared))

# # Load trained parameters
# θθ = load("output/multigrid/theta_cnn_shared.jld2")
# copyto!.(θ_cnn_shared, θθ["theta"])

# Train grid-specialized closure models
θ_cnn = map(dataloaders) do d
    # Prepare training
    θ = T(1.0e-1) * device(θ₀)
    opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
    callbackstate = Point2f[]

    # Training with multiple grids at the same time
    (; opt, θ, callbackstate) = train(
        [d],
        loss,
        opt,
        θ;
        niter = 5000,
        ncallback = 20,
        callbackstate,
        callback = create_callback(closure, device(validset)...; state = callbackstate),
    )
    θ
end
GC.gc()
CUDA.reclaim()

# # Save trained parameters
# jldsave("output/multigrid/theta_cnn.jld2"; theta = Array.(θ_cnn))

# # Load trained parameters
# θθ = load("output/multigrid/theta_cnn.jld2")
# copyto!.(θ_cnn, θθ["theta"])

# Train Smagorinsky closure model
ig = 2;
setup = setups_train[ig];
sample = data_train[1];
m = smagorinsky_closure(setup);
θ = T(0.05)
e_smag = sum(2:length(sample.t)) do it
    It = setup.grid.Ip
    u = sample.u[ig][it] |> device
    c = sample.c[ig][it] |> device
    mu = m(u, θ)
    e = zero(eltype(u[1]))
    for α = 1:D
        # e += sum(abs2, mu[α][Ip] .- c[α][Ip]) / sum(abs2, c[α][Ip])
        e += norm(mu[α][Ip] .- c[α][Ip]) / norm(c[α][Ip])
    end
    e / D
end / length(sample.t)
# for θ in LinRange(T(0), T(1), 100)];

# lines(LinRange(T(0), T(1), 100), e_smag)

# Errors for grid-specialized closure models
e_cnn = zeros(T, length(θ_cnn))
i_traintest = 2:4
offset_i = 1
for (i, setup) in enumerate(setups_test[2:4])
    ig = i + offset_i
    params = params_test[ig]
    pressure_solver = SpectralPressureSolver(setup)
    u = device.(data_test.u[ig])
    u₀ = device(data_test.u[ig][1])
    p₀ = IncompressibleNavierStokes.pressure(pressure_solver, u₀, T(0), setup)
    Δt = params_test.Δt * params_test.savefreq
    tlims = extrema(data_test.t)
    nupdate = 4
    Δt /= nupdate
    processors = (; relerr = relerr_trajectory(u, setup; nupdate))
    closedsetup = (; setup..., closure_model = wrappedclosure(closure, θ_cnn[i], setup))
    _, outputs = solve_unsteady(closedsetup, u₀, p₀, tlims; Δt, pressure_solver, processors)
    e_cnn[i] = outputs.relerr[]
end

# Errors for all test grids
e_nm = zeros(T, length(setups_test))
e_smag = zeros(T, length(setups_test))
e_cnn_shared = zeros(T, length(setups_test))
for (ig, setup) in enumerate(setups_test)
    @show ig
    params = params_test[ig]
    pressure_solver = SpectralPressureSolver(setup)
    u = device.(data_test.u[ig])
    u₀ = device(data_test.u[ig][1])
    p₀ = IncompressibleNavierStokes.pressure(pressure_solver, u₀, T(0), setup)
    Δt = params_test.Δt * params_test.savefreq
    tlims = extrema(data_test.t)
    nupdate = 4
    Δt /= nupdate
    processors = (; relerr = relerr_trajectory(u, setup; nupdate))
    _, outputs = solve_unsteady(setup, u₀, p₀, tlims; Δt, pressure_solver, processors)
    e_nm[ig] = outputs.relerr[]
    m = smagorinsky_closure(setup)
    closedsetup = (; setup..., closure_model = u -> m(u, T(0.1)))
    _, outputs = solve_unsteady(closedsetup, u₀, p₀, tlims; Δt, pressure_solver, processors)
    e_smag[ig] = outputs.relerr[]
    closedsetup = (; setup..., closure_model = wrappedclosure(closure, θ_cnn_shared, setup))
    _, outputs = solve_unsteady(closedsetup, u₀, p₀, tlims; Δt, pressure_solver, processors)
    e_cnn_shared[ig] = outputs.relerr[]
end

GC.gc()
CUDA.reclaim()

e_nm
e_smag
e_cnn
e_cnn_shared
# e_cnn = ones(T, length(setups_test))
# e_fno_shared = ones(T, length(setups_test))
# e_fno_spec = ones(T, length(setups_test))

CairoMakie.activate!()
GLMakie.activate!()

# Plot convergence
with_theme(;
    # linewidth = 5,
    # markersize = 10,
    # markersize = 20,
    # fontsize = 20,
    palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"]),
) do
    nles = params_test.nles
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xticks = nles,
        xlabel = "n",
        title = "Relative error (DNS: n = $(params_test.ndns))",
    )
    scatterlines!(nles, e_nm; marker = :circle, label = "No closure")
    scatterlines!(nles, e_smag; marker = :utriangle, label = "Smagorinsky")
    scatterlines!(nles, e_cnn_shared; marker = :diamond, label = "CNN (shared)")
    scatterlines!(params_train.nles, e_cnn; marker = :rect, label = "CNN (specialized)")
    # scatterlines!(nles, e_fno_spec; label = "FNO (retrained)")
    # scatterlines!(nles, e_fno_share; label = "FNO (shared parameters)")
    lines!(
        collect(extrema(nles[3:end])),
        n -> 2e4 * n^-2.0;
        linestyle = :dash,
        label = "n⁻²",
    )
    axislegend(; position = :lb)
    fig
end

save("convergence.pdf", current_figure())

markers_labels = [
    (:circle, ":circle"),
    (:rect, ":rect"),
    (:diamond, ":diamond"),
    (:hexagon, ":hexagon"),
    (:cross, ":cross"),
    (:xcross, ":xcross"),
    (:utriangle, ":utriangle"),
    (:dtriangle, ":dtriangle"),
    (:ltriangle, ":ltriangle"),
    (:rtriangle, ":rtriangle"),
    (:pentagon, ":pentagon"),
    (:star4, ":star4"),
    (:star5, ":star5"),
    (:star6, ":star6"),
    (:star8, ":star8"),
    (:vline, ":vline"),
    (:hline, ":hline"),
    ('a', "'a'"),
    ('B', "'B'"),
    ('↑', "'\\uparrow'"),
    ('😄', "'\\:smile:'"),
    ('✈', "'\\:airplane:'"),
]

# Final spectra
ig = 4
setup = setups_test[ig];
params = params_test
pressure_solver = SpectralPressureSolver(setup);
uref = device(data_test.u[ig][end]);
u₀ = device(data_test.u[ig][1]);
p₀ = IncompressibleNavierStokes.pressure(pressure_solver, u₀, T(0), setup);
Δt = params_test.Δt * params_test.savefreq;
tlims = extrema(data_test.t);
nupdate = 4;
Δt /= nupdate;
state_nm, outputs = solve_unsteady(setup, u₀, p₀, tlims; Δt, pressure_solver);
m = smagorinsky_closure(setup);
closedsetup = (; setup..., closure_model = u -> m(u, T(0.1)));
state_smag, outputs = solve_unsteady(closedsetup, u₀, p₀, tlims; Δt, pressure_solver);
# closedsetup = (; setup..., closure_model = wrappedclosure(closure, θ_cnn_shared, setup));
closedsetup = (; setup..., closure_model = wrappedclosure(closure, θ_cnn[ig-1], setup));
state_cnn, outputs = solve_unsteady(closedsetup, u₀, p₀, tlims; Δt, pressure_solver);

# Plot predicted spectra
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    D = params.D
    backend = KernelAbstractions.get_backend(setup.grid.x[1])
    K = size(setup.grid.Ip) .÷ 2
    kx = ntuple(α -> 1:K[α]-1, params_train.D)
    k = zero(similar(setup.grid.x[1], length.(kx)))
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)
    # Make averaging matrix
    naverage = 20^D
    i = sortperm(k)
    nbin, r = divrem(length(i), naverage)
    ib = KernelAbstractions.zeros(backend, Int, nbin * naverage)
    ia = KernelAbstractions.zeros(backend, Int, nbin * naverage)
    for j = 1:naverage
        copyto!(view(ia, (j-1)*nbin+1:j*nbin), collect(1:nbin))
        ib[(j-1)*nbin+1:j*nbin] = i[j:naverage:end-r]
    end
    vals = KernelAbstractions.ones(backend, T, nbin * naverage) / naverage
    A = sparse(ia, ib, vals, nbin, length(i))
    k = Array(A * k)
    # Build inertial slope above energy
    @show length(k)
    krange = collect(extrema(k[3:end-5]))
    # slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}") : (-T(5 / 3), L"$k^{-5/3}")
    slope, slopelabel = D == 2 ? (-T(3), "||k||₂⁻³") : (-T(5 / 3), L"k⁻⁵³")
    slopeconst = T(0)
    # Make plot
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xlabel = "||k||₂",
        ylabel = "e(k)",
        # title = "Kinetic energy (n = $(params.nles[ig])) at time t = $(round(data_test.t[end]; digits = 1))",
        title = "Kinetic energy (n = $(params.nles[ig]))",
        xscale = log10,
        yscale = log10,
        xticks = [4, 8, 16, 32, 64, 128],
        limits = (extrema(k)..., T(1e-8), T(1)),
    )
    for (u, label) in (
        # (uref, "Reference"),
        (state_nm.u, "No closure"),
        (state_smag.u, "Smagorinsky"),
        (state_cnn.u, "CNN (specialized)"),
        (uref, "Reference"),
    )
        up = IncompressibleNavierStokes.interpolate_u_p(u, setup)
        e = sum(up -> up[setup.grid.Ip] .^ 2, up)
        ehat = Array(
            A * reshape(abs.(fft(e)[ntuple(α -> kx[α] .+ 1, D)...]) ./ size(e, 1), :),
        )
        slopeconst = max(slopeconst, maximum(ehat ./ k .^ slope))
        lines!(ax, k, ehat; label)
    end
    inertia = 2 .* slopeconst .* krange .^ slope
    lines!(ax, krange, inertia; linestyle = :dash, label = slopelabel)
    axislegend(ax; position = :lb)
    autolimits!(ax)
    fig
end

save("predicted_spectra.pdf", fig)

# Plot spectrum errors
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    D = params.D
    backend = KernelAbstractions.get_backend(setup.grid.x[1])
    K = size(setup.grid.Ip) .÷ 2
    kx = ntuple(α -> 1:K[α]-1, params_train.D)
    k = zero(similar(setup.grid.x[1], length.(kx)))
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)
    # Make averaging matrix
    naverage = 20^D
    i = sortperm(k)
    nbin, r = divrem(length(i), naverage)
    ib = KernelAbstractions.zeros(backend, Int, nbin * naverage)
    ia = KernelAbstractions.zeros(backend, Int, nbin * naverage)
    for j = 1:naverage
        copyto!(view(ia, (j-1)*nbin+1:j*nbin), collect(1:nbin))
        ib[(j-1)*nbin+1:j*nbin] = i[j:naverage:end-r]
    end
    vals = KernelAbstractions.ones(backend, T, nbin * naverage) / naverage
    A = sparse(ia, ib, vals, nbin, length(i))
    k = Array(A * k)
    # Build inertial slope above energy
    @show length(k)
    krange = collect(extrema(k[20:end]))
    # slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}") : (-T(5 / 3), L"$k^{-5/3}")
    slope, slopelabel = D == 2 ? (-T(3), "||k||₂⁻³") : (-T(5 / 3), L"k⁻⁵³")
    slopeconst = T(0)
    # Make plot
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xlabel = "||k||₂",
        ylabel = "e(k)",
        # title = "Kinetic energy (n = $(params.nles[ig])) at time t = $(round(data_test.t[end]; digits = 1))",
        title = "Relative energy error (n = $(params.nles[ig]))",
        xscale = log10,
        yscale = log10,
        xticks = [4, 8, 16, 32, 64, 128],
        limits = (extrema(k)..., T(1e-8), T(1)),
    )
    up = IncompressibleNavierStokes.interpolate_u_p(uref, setup)
    e = sum(up -> up[setup.grid.Ip] .^ 2, up)
    eref =
        Array(A * reshape(abs.(fft(e)[ntuple(α -> kx[α] .+ 1, D)...]) ./ size(e, 1), :))
    for (u, label) in (
        (state_nm.u, "No closure"),
        (state_smag.u, "Smagorinsky"),
        (state_cnn.u, "CNN (specialized)"),
    )
        up = IncompressibleNavierStokes.interpolate_u_p(u, setup)
        e = sum(up -> up[setup.grid.Ip] .^ 2, up)
        ehat = Array(
            A * reshape(abs.(fft(e)[ntuple(α -> kx[α] .+ 1, D)...]) ./ size(e, 1), :),
        )
        ee = @. abs(ehat - eref) / abs(eref)
        lines!(ax, k, ee; label)
    end
    axislegend(ax; position = :lt)
    autolimits!(ax)
    fig
end

save("spectrum_error.pdf", fig)
