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
using CairoMakie
using GLMakie
using IncompressibleNavierStokes
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Random
using Zygote
using SparseArrays
using KernelAbstractions
using FFTW

getorder(i) =
    if i == 1
        :first
    elseif i == 2
        :second
    elseif i == 3
        :last
    else
        error("Unknown order: $i")
    end

GLMakie.activate!()

set_theme!(; GLMakie = (; scalefactor = 1.5))

output = "../SupervisedClosure/figures/"

# Random number generator
rng = Random.default_rng()
Random.seed!(rng, 12345)

# Floating point precision
T = Float64

# Array type
ArrayType = Array
device = identity
clean() = nothing
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using LuxCUDA
using CUDA;
# T = Float64;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);
device = x -> adapt(CuArray, x)
clean() = (GC.gc(); CUDA.reclaim())

# Parameters
get_params(nlesscalar) = (;
    D = 2,
    Re = T(10_000),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(5e-5),
    nles = map(n -> (n, n), nlesscalar),
    # ndns = (n -> (n, n))(1024),
    # ndns = (n -> (n, n))(2048),
    ndns = (n -> (n, n))(4096),
    filters = (FaceAverage(), VolumeAverage()),
    ArrayType,
    PSolver = SpectralPressureSolver,
    icfunc = (setup, psolver) -> random_field(
        setup,
        zero(eltype(setup.grid.x[1]));
        # A = 1,
        kp = 20,
        psolver,
    ),
)

params_train = (; get_params([64, 128, 256])..., savefreq = 10);
params_valid = (; get_params([64, 128, 256])..., tsim = T(0.1), savefreq = 40);
params_test = (; get_params([64, 128, 256, 512, 1024])..., tsim = T(0.2), savefreq = 20);

# Create LES data from DNS
data_train = [create_les_data(; params_train...) for _ = 1:5];
data_valid = [create_les_data(; params_valid...) for _ = 1:1];
data_test = create_les_data(; params_test...);

# Save filtered DNS data
jldsave("output/divfree/data_train.jld2"; data_train)
jldsave("output/divfree/data_valid.jld2"; data_valid)
jldsave("output/divfree/data_test.jld2"; data_test)

# Load filtered DNS data
data_train = load("output/divfree/data.jld2", "data_train");
data_valid = load("output/divfree/data.jld2", "data_valid");
data_test = load("output/divfree/data.jld2", "data_test");

# Build LES setup and assemble operators
getsetups(params) = [
    Setup(
        ntuple(α -> LinRange(T(0), T(1), nles[α] + 1), params.D)...;
        params.Re,
        params.ArrayType,
    ) for nles in params.nles
]
setups_train = getsetups(params_train);
setups_valid = getsetups(params_valid);
setups_test = getsetups(params_test);

data_train[1].t
data_train[1].data |> size
data_train[1].data[1, 1].u[end][1]

# Create input/output arrays
io_train = create_io_arrays(data_train, setups_train);
io_valid = create_io_arrays(data_valid, setups_valid);

# jldsave("output/divfree/io_train.jld2"; io_train)
# jldsave("output/divfree/io_train.jld2"; io_valid)

io_train[1].u |> extrema
io_train[1].c |> extrema
io_valid[1].u |> extrema
io_valid[1].c |> extrema

# Inspect data
let
    ig = 2
    ifil = 1
    field, setup = data_train[1].data[ig, ifil].u, setups_train[ig]
    # field, setup = data_valid[1].data[ig, ifil].u, setups_valid[ig];
    # field, setup = data_test.data[ig, ifil], setups_test[ig];
    u = device.(field[1])
    o = Observable((; u, t = nothing))
    # energy_spectrum_plot(o; setup) |> display
    fieldplot(
        o;
        setup,
        # fieldname = :velocity,
        # fieldname = 1,
    ) |> display
    for i = 1:length(field)
        o[] = (; o[]..., u = device(field[i]))
        sleep(0.001)
    end
end

GLMakie.activate!()
CairoMakie.activate!()

# Training data plot
ifil = 1
boxx = T(0.3), T(0.5)
boxy = T(0.5), T(0.7)
box = [
    Point2f(boxx[1], boxy[1]),
    Point2f(boxx[2], boxy[1]),
    Point2f(boxx[2], boxy[2]),
    Point2f(boxx[1], boxy[2]),
    Point2f(boxx[1], boxy[1]),
]
# fig = with_theme() do
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    sample = data_train[1]
    fig = Figure()
    for (i, it) in enumerate((1, length(sample.t)))
        # for (j, ig) in enumerate((1, 2, 3))
        for (j, ig) in enumerate((1, 2))
            setup = setups_train[ig]
            xf = Array.(getindex.(setup.grid.xp, setup.grid.Ip.indices))
            u = sample.data[ig, ifil].u[it] |> device
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
                limits = (T(0), T(1), T(0), T(1)),
            )
            heatmap!(ax, xf..., ωp; colorrange)
            # lines!(ax, box; color = Cycled(2))
        end
    end
    fig
end

save("$output/training_data.pdf", fig)

closure, θ₀ = cnn(;
    setup = setups_train[1],
    radii = [2, 2, 2, 2],
    channels = [20, 20, 20, params_train.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, false],
    rng,
);
closure.chain

# Train grid-specialized closure models
θ_cnn = map(CartesianIndices(size(io_train))) do I
    # Prepare training
    ig, ifil = I.I
    @show ig ifil
    d = create_dataloader_prior(io_train[ig, ifil]; batchsize = 50, device)
    θ = T(1.0e0) * device(θ₀)
    loss = createloss(mean_squared_error, closure)
    opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
    callbackstate = Point2f[]
    it = rand(1:size(io_valid[ig, ifil].u, 4), 50)
    validset = map(v -> v[:, :, :, it], io_valid[ig, ifil])
    (; opt, θ, callbackstate) = train(
        [d],
        loss,
        opt,
        θ;
        niter = 10000,
        ncallback = 20,
        callbackstate,
        callback = create_relerr_prior(closure, validset),
    )
    θ
end
clean()

# Save trained parameters
jldsave("output/divfree/theta_prior.jld2"; theta = Array.(θ_cnn))

# Load trained parameters
θ_cnn = [device(θ₀) for _ in CartesianIndices(size(data_train[1].data))];
θθ = load("output/divfree/theta_prior.jld2");
copyto!.(θ_cnn, θθ["theta"]);

# θ_cnn_post = let ig = 3, ifil = 2, iorder = 2
θ_cnn_post = map(CartesianIndices((size(io_train)..., 2))) do I
    ig, ifil, iorder = I.I
    println("iorder = $iorder, ifil = $ifil, ig = $ig")
    setup = setups_train[ig]
    psolver = SpectralPressureSolver(setup)
    loss = IncompressibleNavierStokes.create_loss_post(;
        setup,
        psolver,
        closure,
        nupdate = 4,
        projectorder = getorder(iorder),
    )
    data = [(; u = d.data[ig, ifil].u, d.t) for d in data_train]
    d = create_dataloader_post(data; device, nunroll = 20)
    # θ = T(1e-1) * copy(θ_cnn[ig, ifil])
    θ = copy(θ_cnn[ig, ifil])
    # θ = copy(θ_cnn_post)
    # θ = device(θ₀)
    opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
    callbackstate = Point2f[]
    it = 1:20
    data = data_valid[1]
    data = (; u = device.(data.data[ig, ifil].u[it]), t = data.t[it])
    (; opt, θ, callbackstate) = train(
        [d],
        loss,
        opt,
        θ;
        niter = 200,
        ncallback = 10,
        callbackstate,
        callback = create_callback(
            create_relerr_post(;
                data,
                setup,
                psolver,
                closure_model = wrappedclosure(closure, setup),
                projectorder = getorder(iorder),
                nupdate = 8,
            );
            state = callbackstate,
            displayref = false,
        ),
    )
    jldsave(
        "output/divfree/theta_post_iorder$(iorder)_ifil$(ifil)_ig$(ig).jld2";
        theta = Array(θ),
    )
    clean()
    θ
end;
clean()

θ_cnn[2, 2] |> extrema

Array(θ_post)

# Train Smagorinsky model with Lpost (grid search)
θ_smag = map(CartesianIndices((size(io_train, 2), 2))) do I
    ifil, iorder = I.I
    ngrid = size(io_train, 1)
    θmin = T(0)
    emin = T(Inf)
    isample = 1
    it = 1:50
    for θ in LinRange(T(0), T(0.5), 51)
        e = T(0)
        for igrid = 1:ngrid
            println("iorder = $iorder, ifil = $ifil, θ = $θ, igrid = $igrid")
            projectorder = getorder(iorder)
            setup = setups_train[igrid]
            psolver = SpectralPressureSolver(setup)
            d = data_train[isample]
            data = (; u = device.(d.data[igrid, ifil].u[it]), t = d.t[it])
            nupdate = 4
            err = create_relerr_post(;
                data,
                setup,
                psolver,
                closure_model = smagorinsky_closure(setup),
                projectorder,
                nupdate,
            )
            e += err(θ)
        end
        e /= ngrid
        if e < emin
            emin = e
            θmin = θ
        end
    end
    θmin
end
clean()
θ_smag()

# Save trained parameters
jldsave("output/divfree/theta_smag.jld2"; theta = θ_smag);

# Load trained parameters
θ_smag = load("output/divfree/theta_smag.jld2")["theta"];
θ_smag

# lines(LinRange(T(0), T(1), 100), e_smag)

# Errors for grid-specialized closure models
e_nm, e_smag, e_cnn, e_cnn_post = let
    e_nm = zeros(T, size(data_test.data)...)
    e_smag = zeros(T, size(data_test.data)..., 2)
    e_cnn = zeros(T, size(data_test.data)..., 2)
    e_cnn_post = zeros(T, size(data_test.data)..., 2)
    for iorder = 1:2, ifil = 1:2, ig = 1:size(data_test.data, 1)
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        projectorder = getorder(iorder)
        setup = setups_test[ig]
        psolver = SpectralPressureSolver(setup)
        data = (; u = device.(data_test.data[ig, ifil].u), t = data_test.t)
        nupdate = 50
        # No model
        # Only for closurefirst, since projectfirst is the same
        if iorder == 1
            err = create_relerr_post(; data, setup, psolver, closure_model = nothing, nupdate)
            e_nm[ig, ifil] = err(nothing)
        end
        # Smagorinsky
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            closure_model = smagorinsky_closure(setup),
            projectorder,
            nupdate,
        )
        e_smag[ig, ifil, iorder] = err(θ_smag[ifil, iorder])
        # CNN
        # Only the first grids are trained for
        if ig ≤ size(data_train[1].data, 1)
            err = create_relerr_post(;
                data,
                setup,
                psolver,
                closure_model = wrappedclosure(closure, setup),
                projectorder,
                nupdate,
            )
            e_cnn[ig, ifil, iorder] = err(θ_cnn[ig, ifil])
            e_cnn_post[ig, ifil, iorder] = err(θ_cnn_post[ig, ifil, iorder])
        end
    end
    e_nm, e_smag, e_cnn, e_cnn_post
end
clean()
e_nm
e_smag
e_cnn
e_cnn_post

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
    iorder = 1
    lesmodel = iorder == 1 ? "closure-then-project" : "project-then-closure"
    nles = [n[1] for n in params_test.nles]
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xticks = nles,
        xlabel = "Resolution",
        # xlabel = "n",
        # xlabel = L"\bar{n}",
        # title = "Relative error (DNS: n = $(params_test.ndns[1]))",
        title = "Relative error ($lesmodel)",
    )
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "No closure"
        label = label * (ifil == 1 ? " (FA)" : " (VA)")
        scatterlines!(
            nles,
            e_nm[:, ifil];
            color = Cycled(1),
            linestyle,
            marker = :circle,
            label,
        )
    end
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "Smagorinsky"
        label = label * (ifil == 1 ? " (FA)" : " (VA)")
        scatterlines!(
            nles,
            e_smag[:, ifil, iorder];
            color = Cycled(2),
            linestyle,
            marker = :utriangle,
            label,
        )
    end
    for ifil = 1:2
        ntrain = size(data_train[1].data, 1)
        linestyle = ifil == 1 ? :solid : :dash
        label = "CNN"
        label = label * (ifil == 1 ? " (FA)" : " (VA)")
        # ifil == 2 && (label = nothing)
        scatterlines!(
            nles[1:ntrain],
            e_cnn[1:ntrain, ifil, iorder];
            color = Cycled(3),
            linestyle,
            marker = :rect,
            label,
        )
    end
    for ifil = 1:2
        ntrain = size(data_train[1].data, 1)
        linestyle = ifil == 1 ? :solid : :dash
        label = "CNN (post)"
        label = label * (ifil == 1 ? " (FA)" : " (VA)")
        # ifil == 2 && (label = nothing)
        scatterlines!(
            nles[1:ntrain],
            e_cnn_post[1:ntrain, ifil, iorder];
            color = Cycled(4),
            linestyle,
            marker = :diamond,
            label,
        )
    end
    # lines!(
    #     collect(extrema(nles[4:end])),
    #     n -> 2e4 * n^-2.0;
    #     linestyle = :dash,
    #     label = "n⁻²",
    #     color = Cycled(1),
    # )
    axislegend(; position = :rt)
    # iorder == 2 && limits!(ax, (T(60), T(1050)), (T(2e-2), T(1e1)))
    fig
end

save("$output/convergence_Lprior_prosecond.pdf", current_figure())
save("$output/convergence_Lprior_profirst.pdf", current_figure())

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

save("predicted_spectra.pdf", fig)

save("spectrum_error.pdf", fig)
