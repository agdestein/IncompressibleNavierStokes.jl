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
        :last
    elseif i == 3
        :second
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
params_test = (; get_params([64, 128, 256, 512, 1024])..., tsim = T(0.1), savefreq = 10);

# Create LES data from DNS
data_train = [create_les_data(; params_train...) for _ = 1:5];
data_valid = [create_les_data(; params_valid...) for _ = 1:1];
data_test = create_les_data(; params_test...);

# Save filtered DNS data
jldsave("output/divfree/data_train.jld2"; data_train)
jldsave("output/divfree/data_valid.jld2"; data_valid)
jldsave("output/divfree/data_test.jld2"; data_test)

# Load filtered DNS data
data_train = load("output/divfree/data_train.jld2", "data_train");
data_valid = load("output/divfree/data_valid.jld2", "data_valid");
data_test = load("output/divfree/data_test.jld2", "data_test");

data_train[5].comptime
data_valid[1].comptime
data_test.comptime

map(d -> d.comptime, data_train)

sum(d -> d.comptime, data_train) / 60

data_test.comptime / 60

(sum(d -> d.comptime, data_train) + sum(d -> d.comptime, data_valid) + data_test.comptime)

data_train[1].data[1].u[1][1]
data_test.data[6].u[1][1]

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
    # field, setup = data_test.data[ig, ifil].u, setups_test[ig];
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

# Architecture 1
mname="balzac"
closure, θ₀ = cnn(;
    setup = setups_train[1],
    radii = [2, 2, 2, 2],
    channels = [20, 20, 20, params_train.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, false],
    rng,
);
closure.chain

# Architecture 2
mname="rimbaud"
closure, θ₀ = cnn(;
    setup = setups_train[1],
    radii = [2, 2, 2, 2, 2],
    channels = [24, 24, 24, 24, params_train.D],
    # activations = [leakyrelu, leakyrelu, leakyrelu, leakyrelu, identity],
    activations = [tanh, tanh, tanh, tanh, identity],
    use_bias = [true, true, true, true, false],
    rng,
);
closure.chain
mkpath("output/divfree/$mname")

closure(device(io_train[1, 1].u[:, :, :, 1:50]), device(θ₀));

# A-priori training
prior = map(CartesianIndices(size(io_train))) do I
    # Prepare training
    starttime = time()
    ig, ifil = I.I
    println("ig = $ig, ifil = $ifil")
    d = create_dataloader_prior(io_train[ig, ifil]; batchsize = 50, device)
    θ = T(1.0e0) * device(θ₀)
    loss = create_loss_prior(mean_squared_error, closure)
    opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
    it = rand(1:size(io_valid[ig, ifil].u, 4), 50)
    validset = device(map(v -> v[:, :, :, it], io_valid[ig, ifil]))
    (; callbackstate, callback) = create_callback(
        create_relerr_prior(closure, validset...);
        θ,
        displayref = true,
        display_each_iteration = false,
    )
    (; opt, θ, callbackstate) = train(
        [d],
        loss,
        opt,
        θ;
        niter = 10_000,
        ncallback = 20,
        callbackstate,
        callback,
    )
    θ = callbackstate.θmin # Use best θ instead of last θ
    prior = (; θ = Array(θ), comptime = time() - starttime, callbackstate.hist)
    jldsave("output/divfree/$mname/prior_ifilter$(ifil)_igrid$(ig).jld2"; prior)
    prior
end
clean()

# Load trained parameters
prior = map(CartesianIndices(size(io_train))) do I
    ig, ifil = I.I
    name = "output/divfree/$mname/prior_ifilter$(ifil)_igrid$(ig).jld2"
    load(name)["prior"]
end
θ_cnn_prior = [copyto!(device(θ₀), p.θ) for p in prior];

θ_cnn_prior .|> extrema

map(p -> p.comptime, prior)
map(p -> p.comptime, prior) |> vec
map(p -> p.comptime, prior) |> sum
map(p -> p.comptime, prior) |> sum |> x -> x / 60
map(p -> p.comptime, prior) |> sum |> x -> x / 3600

# A-posteriori training ######################################################

let
    ngrid, nfilter = size(io_train)
    for iorder = 1:2, ifil = 1:nfilter, ig = 1:ngrid
        clean()
        starttime = time()
        # (ig, ifil, iorder) == (3, 1, 1) || continue
        # (ifil, iorder) == (1, 1) && continue
        # (ifil, iorder) == (2, 2) || continue
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        setup = setups_train[ig]
        psolver = SpectralPressureSolver(setup)
        loss = IncompressibleNavierStokes.create_loss_post(;
            setup,
            psolver,
            closure,
            nupdate = 2,
            projectorder = getorder(iorder),
        )
        data = [(; u = d.data[ig, ifil].u, d.t) for d in data_train]
        d = create_dataloader_post(data; device, nunroll = 20)
        # θ = T(1e-1) * copy(θ_cnn_prior[ig, ifil])
        θ = copy(θ_cnn_prior[ig, ifil])
        # θ = copy(θ_cnn_post)
        # θ = copy(θ_cnn_post[ig, ifil, iorder])
        # θ = device(θ₀)
        opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
        it = 1:30
        data = data_valid[1]
        data = (; u = device.(data.data[ig, ifil].u[it]), t = data.t[it])
        (; callbackstate, callback) = create_callback(
            create_relerr_post(;
                data,
                setup,
                psolver,
                closure_model = wrappedclosure(closure, setup),
                projectorder = getorder(iorder),
                nupdate = 2,
            );
            θ,
            displayref = false,
        )
        (; opt, θ, callbackstate) =
            train([d], loss, opt, θ; niter = 2000, ncallback = 10, callbackstate, callback)
        θ = callbackstate.θmin # Use best θ instead of last θ
        post = (; θ = Array(θ), comptime = time() - starttime)
        jldsave("output/divfree/$mname/post_iorder$(iorder)_ifil$(ifil)_ig$(ig).jld2"; post)
    end;
    clean()
end

post = map(CartesianIndices((size(io_train)..., 2))) do I
    ig, ifil, iorder = I.I
    name = "output/divfree/$mname/post_iorder$(iorder)_ifil$(ifil)_ig$(ig).jld2"
    load(name)["post"]
end;
θ_cnn_post = [copyto!(device(θ₀), p.θ) for p in post];

θ_cnn_post .|> extrema

map(p -> p.comptime, post)
map(p -> p.comptime, post) |> x -> reshape(x, 6, 2)
map(p -> p.comptime, post) ./ 60
map(p -> p.comptime, post) |> sum
map(p -> p.comptime, post) |> sum |> x -> x / 60
map(p -> p.comptime, post) |> sum |> x -> x / 3600

# Train Smagorinsky model with Lpost (grid search)
smag = map(CartesianIndices((size(io_train, 2), 2))) do I
    starttime = time()
    ifil, iorder = I.I
    ngrid = size(io_train, 1)
    θmin = T(0)
    emin = T(Inf)
    isample = 1
    it = 1:50
    for θ in LinRange(T(0), T(0.5), 501)
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
    (; θ = θmin, comptime = time() - starttime)
end
clean()

smag

# Save trained parameters
jldsave("output/divfree/smag.jld2"; smag);

# Load trained parameters
smag = load("output/divfree/smag.jld2")["smag"];

# Extract coefficients
θ_smag = map(s -> s.θ, smag)

map(s -> s.comptime, smag)
map(s -> s.comptime, smag) |> sum

# lines(LinRange(T(0), T(1), 100), e_smag)

# Compute posterior errors
e_nm, e_smag, e_cnn, e_cnn_post = let
    e_nm = zeros(T, size(data_test.data)...)
    e_smag = zeros(T, size(data_test.data)..., 2)
    e_cnn = zeros(T, size(data_test.data)..., 2)
    e_cnn_post = zeros(T, size(data_test.data)..., 2)
    for iorder = 1:2, ifil = 1:2, ig = 1:size(data_test.data, 1)
        # (ig, ifil, iorder) == (2, 2, 2) || continue
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        projectorder = getorder(iorder)
        setup = setups_test[ig]
        psolver = SpectralPressureSolver(setup)
        data = (; u = device.(data_test.data[ig, ifil].u), t = data_test.t)
        # nupdate = ig > 3 ? 4 : 2
        nupdate = 2
        # No model
        # Only for closurefirst, since projectfirst is the same
        if iorder == 2
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
                # nupdate = 50,
            )
            e_cnn[ig, ifil, iorder] = err(θ_cnn_prior[ig, ifil])
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

data_train[1].t[2] - data_train[1].t[1]
data_test.t[2] - data_test.t[1]

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
    iorder = 2
    # lesmodel = iorder == 1 ? "project-then-closure" : "closure-then-project"
    # lesmodel = iorder == 1 ? "project-then-closure" : "closure-then-project"
    lesmodel = iorder == 1 ? "Gen" : "DFC"
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
        # label = label * (ifil == 1 ? " (FA)" : " (VA)")
        ifil == 2 && (label = nothing)
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
        # label = label * (ifil == 1 ? " (FA)" : " (VA)")
        ifil == 2 && (label = nothing)
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
        label = "CNN (prior)"
        # label = label * (ifil == 1 ? " (FA)" : " (VA)")
        ifil == 2 && (label = nothing)
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
        # label = label * (ifil == 1 ? " (FA)" : " (VA)")
        ifil == 2 && (label = nothing)
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

name = "$output/convergence"
ispath(name) || mkpath(name)
save("$name/$(mname)_gen.pdf", current_figure())
save("$name/$(mname)_dfc.pdf", current_figure())

# Energy evolution ###########################################################

kineticenergy = let
    clean()
    ngrid, nfilter = size(io_train)
    ke_ref = fill(zeros(T, 0), ngrid, nfilter)
    ke_nomodel = fill(zeros(T, 0), ngrid, nfilter)
    ke_smag = fill(zeros(T, 0), ngrid, nfilter, 2)
    ke_cnn_prior = fill(zeros(T, 0), ngrid, nfilter, 2)
    ke_cnn_post = fill(zeros(T, 0), ngrid, nfilter, 2)
    for iorder = 1:2, ifil = 1:nfilter, ig = 1:ngrid
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        setup = setups_test[ig]
        psolver = SpectralPressureSolver(setup)
        t = data_test.t
        u₀ = data_test.data[ig, ifil].u[1] |> device
        tlims = (t[1], t[end])
        # nupdate = 50
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(u₀[1])
        ewriter = processor() do state
            ehist = zeros(T, 0)
            on(state) do (; u, n)
                if n % nupdate == 0
                    e = IncompressibleNavierStokes.total_kinetic_energy(u, setup)
                    push!(ehist, e)
                end
            end
            state[] = state[] # Compute initial energy
            ehist
        end
        processors = (; ewriter)
        if iorder == 1
            # Does not depend on projection order
            ke_ref[ig, ifil] = map(
                u -> IncompressibleNavierStokes.total_kinetic_energy(device(u), setup),
                data_test.data[ig, ifil].u,
            )
            ke_nomodel[ig, ifil] =
                solve_unsteady(setup, u₀, tlims; Δt, processors, psolver)[2].ewriter
        end
        ke_smag[ig, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = smagorinsky_closure(setup),
                ),
                u₀,
                tlims;
                Δt,
                processors,
                psolver,
                θ = θ_smag[ifil, iorder],
            )[2].ewriter
        ke_cnn_prior[ig, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                u₀,
                tlims;
                Δt,
                processors,
                psolver,
                θ = θ_cnn_prior[ig, ifil],
            )[2].ewriter
        ke_cnn_post[ig, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                u₀,
                tlims;
                Δt,
                processors,
                psolver,
                θ = θ_cnn_post[ig, ifil, iorder],
            )[2].ewriter
    end
    (; ke_ref, ke_nomodel, ke_smag, ke_cnn_prior, ke_cnn_post)
end;
clean();

kineticenergy.ke_ref[1]
kineticenergy.ke_nomodel[1]
kineticenergy.ke_smag[1]
kineticenergy.ke_cnn_prior[1]
kineticenergy.ke_cnn_post[1]

CairoMakie.activate!()

with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    # t = data_test.t[2:end]
    t = data_test.t
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        # reflevel = kineticenergy.ke_ref[igrid, ifil][2:end]
        reflevel = copy(kineticenergy.ke_ref[igrid, ifil])
        reflevel = fill!(reflevel, 1)
        lesmodel = iorder == 1 ? "Gen" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            xlabel = "t",
            ylabel = "E(t)",
            # title = "Kinetic energy: $lesmodel, $fil,  $nles",
            title = "Kinetic energy: $lesmodel, $fil",
        )
        lines!(
            ax,
            t,
            # kineticenergy.ke_ref[igrid, ifil][2:end] ./ reflevel;
            kineticenergy.ke_ref[igrid, ifil] ./ reflevel;
            color = Cycled(1),
            linestyle = :dash,
            label = "Reference",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_nomodel[igrid, ifil] ./ reflevel;
            color = Cycled(1),
            label = "No closure",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_smag[igrid, ifil, iorder] ./ reflevel;
            color = Cycled(2),
            label = "Smagorinsky",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_cnn_prior[igrid, ifil, iorder] ./ reflevel;
            color = Cycled(3),
            label = "CNN (prior)",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_cnn_post[igrid, ifil, iorder] ./ reflevel;
            color = Cycled(4),
            label = "CNN (post)",
        )
        iorder == 1 && axislegend(; position = :lt)
        iorder == 2 && axislegend(; position = :lb)
        # axislegend(; position = :lb)
        # axislegend()
        name = "$output/energy_evolution/$mname/"
        ispath(name) || mkpath(name)
        save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
    end
end

# Divergence #################################################################

divs = let
    clean()
    ngrid, nfilter = size(io_train)
    d_ref = fill(zeros(T, 0), ngrid, nfilter)
    d_nomodel = fill(zeros(T, 0), ngrid, nfilter, 3)
    d_smag = fill(zeros(T, 0), ngrid, nfilter, 3)
    d_cnn_prior = fill(zeros(T, 0), ngrid, nfilter, 3)
    d_cnn_post = fill(zeros(T, 0), ngrid, nfilter, 3)
    for iorder = 1:3, ifil = 1:nfilter, ig = 1:ngrid
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        setup = setups_test[ig]
        psolver = SpectralPressureSolver(setup)
        t = data_test.t
        u₀ = data_test.data[ig, ifil].u[1] |> device
        tlims = (t[1], t[end])
        # nupdate = 50
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(u₀[1])
        dwriter = processor() do state
            div = fill!(similar(setup.grid.x[1], setup.grid.N), 0)
            dhist = zeros(T, 0)
            on(state) do (; u, n)
                if n % nupdate == 0
                    IncompressibleNavierStokes.divergence!(div, u, setup)
                    d = view(div, setup.grid.Ip)
                    d = sum(abs2, d) / length(d)
                    d = sqrt(d)
                    push!(dhist, d)
                end
            end
            state[] = state[] # Compute initial divergence
            dhist
        end
        processors = (; dwriter)
        if iorder == 1
            # Does not depend on projection order
            d_ref[ig, ifil] = map(data_test.data[ig, ifil].u) do u
                    u = device(u)
                    div = IncompressibleNavierStokes.divergence(u, setup)
                    d = view(div, setup.grid.Ip)
                    d = sum(abs2, d) / length(d)
                    d = sqrt(d)
                end
        end
        iorder_use = iorder == 3 ? 2 : iorder
        d_nomodel[ig, ifil, iorder] =
            solve_unsteady(
                (
                    setup...,
                    projectorder = getorder(iorder),
                ),
                u₀, tlims; Δt, processors, psolver)[2].dwriter
        d_smag[ig, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = smagorinsky_closure(setup),
                ),
                u₀,
                tlims;
                Δt,
                processors,
                psolver,
                θ = θ_smag[ifil, iorder_use],
            )[2].dwriter
        d_cnn_prior[ig, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                u₀,
                tlims;
                Δt,
                processors,
                psolver,
                θ = θ_cnn_prior[ig, ifil],
            )[2].dwriter
        d_cnn_post[ig, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                u₀,
                tlims;
                Δt,
                processors,
                psolver,
                θ = θ_cnn_post[ig, ifil, iorder_use],
            )[2].dwriter
    end
    (;
        d_ref,
        d_nomodel,
        d_smag,
        d_cnn_prior,
        d_cnn_post)
end;
clean();

# Save
jldsave("output/divfree/$(mname)_divs.jld2"; divs)

# Load
divs = load("output/divfree/$(mname)_divs.jld2")["divs"]

divs.d_ref .|> extrema
divs.d_nomodel .|> extrema
divs.d_smag .|> extrema
divs.d_cnn_prior .|> extrema
divs.d_cnn_post .|> extrema

divs.d_ref[1, 1] |> lines
divs.d_ref[1, 2] |> lines!

divs.d_nomodel[1, 1] |> lines
divs.d_nomodel[1, 2] |> lines!

divs.d_cnn_post[1, 1, 3] |> lines
divs.d_cnn_post[1, 2, 3] |> lines!

divs.d_smag[1, 2, 3]
divs.d_smag[1, 1, 3]
divs.d_smag[1, 2, 3]

CairoMakie.activate!()

with_theme(; 
    fontsize = 20,
    palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"]),
) do
    t = data_test.t
    for islog in (true, false)
    for iorder = 1:3, ifil = 1:2, igrid = 1:3
        # println("iorder = $iorder, igrid = $igrid")
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        lesmodel = 
            if iorder == 1 
                "Gen"
            elseif iorder == 2 
                "DCF"
            else 
                "DCF-RHS"
            end
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        fig = Figure(; size = (500, 400))
        yscale = islog ? log10 : identity
        ax = Axis(
            fig[1, 1];
            yscale,
            xlabel = "t",
            # ylabel = "Dv",
            # title = "Divergence: $lesmodel, $nles",
            title = "Divergence: $lesmodel, $fil,  $nles",
            # title = "Divergence: $lesmodel, $fil",
        )
        linestyle = ifil == 1 ? :solid : :dash
        lines!(
            ax,
            t,
            divs.d_ref[igrid, ifil];
            color = Cycled(1),
            linestyle = :dash,
            label = "Reference",
        )
        lines!(
            ax,
            t,
            divs.d_nomodel[igrid, ifil, iorder];
            color = Cycled(1),
            label = "No closure",
        )
        lines!(
            ax,
            t,
            divs.d_smag[igrid, ifil, iorder];
            color = Cycled(2),
            label = "Smagorinsky",
        )
        lines!(
            ax,
            t,
            divs.d_cnn_prior[igrid, ifil, iorder];
            color = Cycled(3),
            label = "CNN (prior)",
        )
        lines!(
            ax,
            t,
            divs.d_cnn_post[igrid, ifil, iorder];
            color = Cycled(4),
            label = "CNN (post)",
        )
        # axislegend()
        # iorder == 1 && axislegend(; position = :lt)
        # iorder == 2 && axislegend(; position = :lb)
        iorder == 2 && ifil == 1 && axislegend(; position = :rt)
        # axislegend()
        islog && ylims!(ax, (T(1e-6), T(1e3)))
        name = "$output/divergence/$mname/$(islog ? "log" : "lin")"
        ispath(name) || mkpath(name)
        # save("$(name)/iorder$(iorder)_igrid$(igrid).pdf", fig)
        save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
    end
    end
end

# Solutions at final time ####################################################

ufinal = let
    ngrid, nfilter = size(io_train)
    temp = ntuple(α -> zeros(T, 0, 0), 2)
    u_ref = fill(temp, ngrid, nfilter)
    u_nomodel = fill(temp, ngrid, nfilter)
    u_smag = fill(temp, ngrid, nfilter, 2)
    u_cnn_prior = fill(temp, ngrid, nfilter, 2)
    u_cnn_post = fill(temp, ngrid, nfilter, 2)
    for iorder = 1:2, ifil = 1:nfilter, igrid = 1:ngrid
        clean()
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        t = data_test.t
        setup = setups_test[igrid]
        psolver = SpectralPressureSolver(setup)
        u₀ = data_test.data[igrid, ifil].u[1] |> device
        tlims = (t[1], t[end])
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(u₀[1])
        if iorder == 1
            # Does not depend on projection order
            u_ref[igrid, ifil] = data_test.data[igrid, ifil].u[end]
            u_nomodel[igrid, ifil] =
                solve_unsteady(setup, u₀, tlims; Δt, psolver)[1].u .|> Array
        end
        u_smag[igrid, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = smagorinsky_closure(setup),
                ),
                u₀,
                tlims;
                Δt,
                psolver,
                θ = θ_smag[ifil, iorder],
            )[1].u .|> Array
        u_cnn_prior[igrid, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                u₀,
                tlims;
                Δt,
                psolver,
                θ = θ_cnn_prior[igrid, ifil],
            )[1].u .|> Array
        u_cnn_post[igrid, ifil, iorder] =
            solve_unsteady(
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                u₀,
                tlims;
                Δt,
                psolver,
                θ = θ_cnn_post[igrid, ifil, iorder],
            )[1].u .|> Array
    end
    (; u_ref, u_nomodel, u_smag, u_cnn_prior, u_cnn_post)
end;
clean();

ufinal.u_ref[1][2]
ufinal.u_cnn_prior[3, 1, 1][1]
ufinal.u_cnn_post[3, 1, 1][1]

jldsave("output/divfree/ufinal_$mname.jld2"; ufinal)

ufinal = load("output/divfree/ufinal_$mname.jld2")["ufinal"];

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

# Plot spectra ###############################################################

fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        lesmodel = iorder == 1 ? "Gen" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        setup = setups_test[igrid]
        fields =
            [
                ufinal.u_ref[igrid, ifil],
                ufinal.u_nomodel[igrid, ifil],
                ufinal.u_smag[igrid, ifil, iorder],
                ufinal.u_cnn_prior[igrid, ifil, iorder],
                ufinal.u_cnn_post[igrid, ifil, iorder],
            ] .|> device
        (; Ip) = setup.grid
        (; A, κ, K) = IncompressibleNavierStokes.spectral_stuff(setup)
        specs = map(fields) do u
            # up = interpolate_u_p(u, setup)
            up = u
            e = sum(up) do u
                u = u[Ip]
                uhat = fft(u)[ntuple(α -> 1:K[α], 2)...]
                # abs2.(uhat)
                abs2.(uhat) ./ (2 * prod(size(u))^2)
                # abs2.(uhat) ./ size(u, 1)
            end
            e = A * reshape(e, :)
            # e = max.(e, eps(T)) # Avoid log(0)
            ehat = Array(e)
        end
        kmax = maximum(κ)
        # Build inertial slope above energy
        krange = [T(16), T(κ[end])]
        slope, slopelabel = -T(3), L"$\kappa^{-3}"
        slopeconst = maximum(specs[1] ./ κ .^ slope)
        offset = 3
        inertia = offset .* slopeconst .* krange .^ slope
        # Nice ticks
        logmax = round(Int, log2(kmax + 1))
        xticks = T(2) .^ (0:logmax)
        # Make plot
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            xticks,
            # xlabel = "k",
            xlabel = "κ",
            # ylabel = "e(κ)",
            xscale = log10,
            yscale = log10,
            limits = (1, kmax, T(1e-8), T(1)),
            title = "Kinetic energy: $lesmodel, $fil",
        )
        lines!(ax, κ, specs[2]; color = Cycled(1), label = "No model")
        lines!(ax, κ, specs[3]; color = Cycled(2), label = "Smagorinsky")
        lines!(ax, κ, specs[4]; color = Cycled(3), label = "CNN (prior)")
        lines!(ax, κ, specs[5]; color = Cycled(4), label = "CNN (post)")
        lines!(ax, κ, specs[1]; color = Cycled(1), linestyle = :dash, label = "Reference")
        lines!(ax, krange, inertia; color = Cycled(1), label = slopelabel, linestyle = :dot)
        # axislegend(ax; position = :lb)
        axislegend(ax; position = :cb)
        autolimits!(ax)
        # limits!(ax, (T(0.8), T(800)), (T(1e-10), T(1)))
        name = "$output/energy_spectra/$mname"
        ispath(name) || mkpath(name)
        save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
    end
end
clean();

# Plot fields ################################################################

GLMakie.activate!()

with_theme(;
    fontsize = 25,
    palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"]),
) do
    x1 = 0.3
    x2 = 0.5
    y1 = 0.5
    y2 = 0.7
    box = [
        Point2f(x1, y1),
        Point2f(x2, y1),
        Point2f(x2, y2),
        Point2f(x1, y2),
        Point2f(x1, y1),
        # Point2f(x2, y1),
    ]
    path = "$output/les_fields/$mname"
    ispath(path) || mkpath(path)
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        setup = setups_test[igrid]
        name = "$path/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid)"
        lesmodel = iorder == 1 ? "Gen" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        function makeplot(u, title, suffix)
            fig = fieldplot(
                (; u, t = T(0));
                setup,
                title,
                # type = image,
                # colormap = :viridis,
                docolorbar = false,
                size = (500, 500),
            )
            lines!(box; linewidth = 5, color = Cycled(2))
            fname = "$(name)_$(suffix).png"
            save(fname, fig)
            # run(`convert $fname -trim $fname`) # Requires imagemagick
        end
        iorder == 2 &&
            makeplot(device(ufinal.u_ref[igrid, ifil]), "Reference, $fil, $nles", "ref")
        iorder == 2 && makeplot(
            device(ufinal.u_nomodel[igrid, ifil]),
            "No closure, $fil, $nles",
            "nomodel",
        )
        makeplot(
            device(ufinal.u_smag[igrid, ifil, iorder]),
            "Smagorinsky, $lesmodel, $fil, $nles",
            "smag",
        )
        makeplot(
            device(ufinal.u_cnn_prior[igrid, ifil, iorder]),
            "CNN (prior), $lesmodel, $fil, $nles",
            "prior",
        )
        makeplot(
            device(ufinal.u_cnn_post[igrid, ifil, iorder]),
            "CNN (post), $lesmodel, $fil, $nles",
            "post",
        )
    end
end
