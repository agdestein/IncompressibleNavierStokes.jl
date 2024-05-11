# # A-posteriori analysis: Large Eddy Simulation (2D)
#
# Generate filtered DNS data, train closure model, compare filters,
# closure models, and projection orders.
# 
# The filtered DNS data is saved and can be loaded in a subesequent session.
# The learned CNN parameters are also saved.

using Adapt
using GLMakie
using CairoMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes.RKMethods
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using NeuralClosure
using NNlib
using Optimisers
using Random
using SparseArrays
using FFTW

# Color palette for consistent theme throughout paper
palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])

# Encode projection order ("close first, then project" etc)
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

# Choose where to put output
plotdir = "output/postanalysis/plots"
outdir = "output/postanalysis"
ispath(plotdir) || mkpath(plotdir)
ispath(outdir) || mkpath(outdir)

# Random number generator seeds ################################################
#
# Use a new RNG with deterministic seed for each code "section"
# so that e.g. training batch selection does not depend on whether we
# generated fresh filtered DNS data or loaded existing one (the
# generation of which would change the state of a global RNG).
#
# Note: Using `rng = Random.default_rng()` twice seems to point to the
# same RNG, and mutating one also mutates the other.
# `rng = Random.Xoshiro()` creates an independent copy each time.
#
# We define all the seeds here so that we don't accidentally type the same seed
# twice.

seeds = (;
    dns = 123, # Initial conditions
    θ₀ = 234, # Initial CNN parameters
    prior = 345, # A-priori training batch selection
    post = 456, # A-posteriori training batch selection
)

# Hardware selection ########################################################

# For running on CPU.
# Consider reducing the sizes of DNS, LES, and CNN layers if
# you want to test run on a laptop.
T = Float32
ArrayType = Array
device = identity
clean() = nothing

# For running on a CUDA compatible GPU
using LuxCUDA
using CUDA;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);
device = x -> adapt(CuArray, x)
clean() = (GC.gc(); CUDA.reclaim())

# Data generation ###########################################################
#
# Create filtered DNS data for training, validation, and testing.

# Random number generator for initial conditions.
# Important: Created and seeded first, then shared for all initial conditions.
# After each initial condition generation, it is mutated and creates different
# IC for the next iteration.
rng = Random.Xoshiro()
Random.seed!(rng, seeds.dns)

# Parameters
get_params(nlesscalar) = (;
    D = 2,
    Re = T(10_000),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(5e-5),
    nles = map(n -> (n, n), nlesscalar), # LES resolutions
    ndns = (n -> (n, n))(4096), # DNS resolution
    filters = (FaceAverage(), VolumeAverage()),
    ArrayType,
    create_psolver = psolver_spectral,
    icfunc = (setup, psolver) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver),
    rng,
)

# Get parameters for multiple LES resolutions
params_train = (; get_params([64, 128, 256])..., tsim = T(0.5), savefreq = 10);
params_valid = (; get_params([64, 128, 256])..., tsim = T(0.1), savefreq = 40);
params_test = (; get_params([64, 128, 256, 512, 1024])..., tsim = T(0.1), savefreq = 10);

# Create filtered DNS data
data_train = [create_les_data(; params_train...) for _ = 1:5];
data_valid = [create_les_data(; params_valid...) for _ = 1:1];
data_test = create_les_data(; params_test...);

# Save filtered DNS data
jldsave("$outdir/data_train.jld2"; data_train)
jldsave("$outdir/data_valid.jld2"; data_valid)
jldsave("$outdir/data_test.jld2"; data_test)

# Load filtered DNS data
data_train = load("$outdir/data_train.jld2", "data_train");
data_valid = load("$outdir/data_valid.jld2", "data_valid");
data_test = load("$outdir/data_test.jld2", "data_test");

# Computational time
data_train[5].comptime
data_valid[1].comptime
data_test.comptime
map(d -> d.comptime, data_train)
sum(d -> d.comptime, data_train) / 60
data_test.comptime / 60
(sum(d -> d.comptime, data_train) + sum(d -> d.comptime, data_valid) + data_test.comptime)

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

# Example data inspection
data_train[1].t
data_train[1].data |> size
data_train[1].data[1, 1].u[end][1]

# Create input/output arrays for a-priori training (ubar vs c)
io_train = create_io_arrays(data_train, setups_train);
io_valid = create_io_arrays(data_valid, setups_valid);
io_test = create_io_arrays([data_test], setups_test);

# # Save IO arrays
# jldsave("$outdir/io_train.jld2"; io_train)
# jldsave("$outdir/io_valid.jld2"; io_valid)
# jldsave("$outdir/io_test.jld2"; io_test)
#
# # Load IO arrays
# io_train = load("$outdir/io_train.jld2"; "io_train")
# io_valid = load("$outdir/io_valid.jld2"; "io_valid")
# io_test = load("$outdir/io_test.jld2"; "io_test")

# Check that data is reasonably bounded
io_train[1].u |> extrema
io_train[1].c |> extrema
io_valid[1].u |> extrema
io_valid[1].c |> extrema
io_test[1].u |> extrema
io_test[1].c |> extrema

# Inspect data (live animation with GLMakie)
GLMakie.activate!()
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
        # fieldname = :velocitynorm,
        # fieldname = 1,
    ) |> display
    for i = 1:length(field)
        o[] = (; o[]..., u = device(field[i]))
        sleep(0.001)
    end
end

# CNN closure model ##########################################################

# Random number generator for initial CNN parameters.
# All training sessions will start from the same θ₀
# for a fair comparison.
rng = Random.Xoshiro()
Random.seed!(rng, seeds.θ₀)

# # CNN architecture 1
# mname = "balzac"
# closure, θ₀ = cnn(;
#     setup = setups_train[1],
#     radii = [2, 2, 2, 2],
#     channels = [20, 20, 20, params_train.D],
#     activations = [leakyrelu, leakyrelu, leakyrelu, identity],
#     use_bias = [true, true, true, false],
#     rng,
# );
# closure.chain

# CNN architecture 2
mname = "rimbaud"
closure, θ₀ = cnn(;
    setup = setups_train[1],
    radii = [2, 2, 2, 2, 2],
    channels = [24, 24, 24, 24, params_train.D],
    activations = [tanh, tanh, tanh, tanh, identity],
    use_bias = [true, true, true, true, false],
    rng,
);
closure.chain

# Save-path for CNN
savepath = "$outdir/$mname"
ispath(savepath) || mkpath(savepath)

# Give the CNN a test run
# Note: Data and parameters are stored on the CPU, and
# must be moved to the GPU before running (`device`)
closure(device(io_train[1, 1].u[:, :, :, 1:50]), device(θ₀));

# A-priori training ###########################################################
#
# Train one set of CNN parameters for each of the filter types and grid sizes.
# Save parameters to disk after each run.
# Plot training progress (for a validation data batch).

priornames = map(CartesianIndices(io_train)) do I
    ig, ifil = I.I
    "$savepath/prior_ifilter$(ifil)_igrid$(ig).jld2"
end

# Train
let
    # Random number generator for batch selection
    rng = Random.Xoshiro()
    Random.seed!(rng, seeds.prior)
    ngrid, nfilter = size(io_train)
    for ifil = 1:nfilter, ig = 1:ngrid
        clean()
        starttime = time()
        println("ig = $ig, ifil = $ifil")
        d = create_dataloader_prior(io_train[ig, ifil]; batchsize = 50, device, rng)
        θ = T(1.0e0) * device(θ₀)
        loss = create_loss_prior(mean_squared_error, closure)
        opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
        it = rand(rng, 1:size(io_valid[ig, ifil].u, 4), 50)
        validset = device(map(v -> v[:, :, :, it], io_valid[ig, ifil]))
        (; callbackstate, callback) = create_callback(
            create_relerr_prior(closure, validset...);
            θ,
            displayref = true,
            display_each_iteration = false, # Set to `true` if using CairoMakie
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
        jldsave(priorfiles[ig, ifil]; prior)
    end
    clean()
end

# Load learned parameters and training times
prior = map(f -> load(f)["prior"], priorfiles)
θ_cnn_prior = [copyto!(device(θ₀), p.θ) for p in prior];

# Check that parameters are within reasonable bounds
θ_cnn_prior .|> extrema

# Training times
map(p -> p.comptime, prior)
map(p -> p.comptime, prior) |> vec
map(p -> p.comptime, prior) |> sum # Seconds
map(p -> p.comptime, prior) |> sum |> x -> x / 60 # Minutes
map(p -> p.comptime, prior) |> sum |> x -> x / 3600 # Hours

# A-posteriori training ######################################################
#
# Train one set of CNN parameters for each
# projection order, filter type and grid size.
# Save parameters to disk after each combination.
# Plot training progress (for a validation data batch).
#
# The time stepper `RKProject` allows for choosing when to project.

postfiles = map(CartesianIndices((size(io_train)..., 2))) do I
    ig, ifil, iorder = I.I
    "$savepath/post_iorder$(iorder)_ifil$(ifil)_ig$(ig).jld2"
end

# Train
let
    # Random number generator for batch selection
    rng = Random.Xoshiro()
    Random.seed!(rng, seeds.post)
    ngrid, nfilter = size(io_train)
    for iorder = 1:2, ifil = 1:nfilter, ig = 1:ngrid
        clean()
        starttime = time()
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        setup = setups_train[ig]
        psolver = psolver_spectral(setup)
        loss = IncompressibleNavierStokes.create_loss_post(;
            setup,
            psolver,
            method = RKProject(RK44(; T), getorder(iorder)),
            closure,
            nupdate = 2, # Time steps per loss evaluation
        )
        data = [(; u = d.data[ig, ifil].u, d.t) for d in data_train]
        d = create_dataloader_post(data; device, nunroll = 20, rng)
        θ = copy(θ_cnn_prior[ig, ifil])
        opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
        it = 1:30
        data = data_valid[1]
        data = (; u = device.(data.data[ig, ifil].u[it]), t = data.t[it])
        (; callbackstate, callback) = create_callback(
            create_relerr_post(;
                data,
                setup,
                psolver,
                method = RKProject(RK44(; T), getorder(iorder)),
                closure_model = wrappedclosure(closure, setup),
                nupdate = 2,
            );
            θ,
            displayref = false,
        )
        (; opt, θ, callbackstate) =
            train([d], loss, opt, θ; niter = 2000, ncallback = 10, callbackstate, callback)
        θ = callbackstate.θmin # Use best θ instead of last θ
        post = (; θ = Array(θ), comptime = time() - starttime)
        jldsave(postfiles[iorder, ifil, ig]; post)
    end
    clean()
end

# Load learned parameters and training times
post = map(f -> load(f)["post"], postfiles);
θ_cnn_post = [copyto!(device(θ₀), p.θ) for p in post];

# Check that parameters are within reasonable bounds
θ_cnn_post .|> extrema

# Training times
map(p -> p.comptime, post)
map(p -> p.comptime, post) |> x -> reshape(x, 6, 2)
map(p -> p.comptime, post) ./ 60
map(p -> p.comptime, post) |> sum
map(p -> p.comptime, post) |> sum |> x -> x / 60
map(p -> p.comptime, post) |> sum |> x -> x / 3600

# Train Smagorinsky model ####################################################
#
# Use a-posteriori error grid search to determine
# the optimal Smagorinsky constant.
# Find one constant for each projection order and filter type. but
# The constant is shared for all grid sizes, since the filter
# width (=grid size) is part of the model definition separately.

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
            psolver = psolver_spectral(setup)
            d = data_train[isample]
            data = (; u = device.(d.data[igrid, ifil].u[it]), t = d.t[it])
            nupdate = 4
            err = create_relerr_post(;
                data,
                setup,
                psolver,
                method = RKProject(RK44(; T), getorder(iorder)),
                closure_model = smagorinsky_closure(setup),
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
jldsave("$outdir/smag.jld2"; smag);

# Load trained parameters
smag = load("$outdir/smag.jld2")["smag"];

# Extract coefficients
θ_smag = map(s -> s.θ, smag)

# Computational time
map(s -> s.comptime, smag)
map(s -> s.comptime, smag) |> sum

# Compute a-priori errors ###################################################
#
# Note that it is still interesting to compute the a-priori errors for the
# a-posteriori trained CNN.

eprior = let
    prior = zeros(T, 3, 2)
    post = zeros(T, 3, 2, 2)
    for ig = 1:3, ifil = 1:2
        println("ig = $ig, ifil = $ifil")
        testset = device(io_test[ig, ifil])
        err = create_relerr_prior(closure, testset...)
        prior[ig, ifil] = err(θ_cnn_prior[ig, ifil])
        for iorder = 1:2
            post[ig, ifil, iorder] = err(θ_cnn_post[ig, ifil, iorder])
        end
    end
    (; prior, post)
end
clean()

eprior.prior
eprior.post

eprior.prior |> x -> reshape(x, :) |> x -> round.(x; digits = 2)
eprior.post |> x -> reshape(x, :, 2) |> x -> round.(x; digits = 2)

# Compute a-posteriori errors #################################################

(; e_nm, e_smag, e_cnn, e_cnn_post) = let
    e_nm = zeros(T, size(data_test.data)...)
    e_smag = zeros(T, size(data_test.data)..., 2)
    e_cnn = zeros(T, size(data_test.data)..., 2)
    e_cnn_post = zeros(T, size(data_test.data)..., 2)
    for iorder = 1:2, ifil = 1:2, ig = 1:size(data_test.data, 1)
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        projectorder = getorder(iorder)
        setup = setups_test[ig]
        psolver = psolver_spectral(setup)
        data = (; u = device.(data_test.data[ig, ifil].u), t = data_test.t)
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
            method = RKProject(RK44(; T), getorder(iorder)),
            closure_model = smagorinsky_closure(setup),
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
                method = RKProject(RK44(; T), getorder(iorder)),
                closure_model = wrappedclosure(closure, setup),
                nupdate,
            )
            e_cnn[ig, ifil, iorder] = err(θ_cnn_prior[ig, ifil])
            e_cnn_post[ig, ifil, iorder] = err(θ_cnn_post[ig, ifil, iorder])
        end
    end
    (; e_nm, e_smag, e_cnn, e_cnn_post)
end
clean()

round.(
    [e_nm[:] reshape(e_smag, :, 2) reshape(e_cnn, :, 2) reshape(e_cnn_post, :, 2)][
        [1:3; 6:8],
        :,
    ];
    sigdigits = 2,
)

# Plot a-priori errors ########################################################

# Better for PDF export
CairoMakie.activate!()

fig = with_theme(; palette) do
    nles = [n[1] for n in params_test.nles][1:3]
    ifil = 1
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        xticks = nles,
        xlabel = "Resolution",
        title = "Relative a-priori error $(ifil == 1 ? " (FA)" : " (VA)")",
    )
    linestyle = :solid
    label = "No closure"
    scatterlines!(
        nles,
        ones(T, length(nles));
        color = Cycled(1),
        linestyle,
        marker = :circle,
        label,
    )
    label = "CNN (Lprior)"
    scatterlines!(
        nles,
        eprior.prior[:, ifil];
        color = Cycled(2),
        linestyle,
        marker = :utriangle,
        label,
    )
    label = "CNN (Lpost, DIF)"
    scatterlines!(
        nles,
        eprior.post[:, ifil, 1];
        color = Cycled(3),
        linestyle,
        marker = :rect,
        label,
    )
    label = "CNN (Lpost, DCF)"
    scatterlines!(
        nles,
        eprior.post[:, ifil, 2];
        color = Cycled(4),
        linestyle,
        marker = :diamond,
        label,
    )
    axislegend(; position = :lb)
    ylims!(ax, (T(-0.05), T(1.05)))
    name = "$plotdir/convergence"
    ispath(name) || mkpath(name)
    save("$name/$(mname)_prior_ifilter$ifil.pdf", fig)
    fig
end

# Plot a-posteriori errors ###################################################

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    iorder = 2
    lesmodel = iorder == 1 ? "DIF" : "DCF"
    ntrain = size(data_train[1].data, 1)
    nles = [n[1] for n in params_test.nles][1:ntrain]
    fig = Figure(; size = (500, 400))
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xticks = nles,
        xlabel = "Resolution",
        title = "Relative error ($lesmodel)",
    )
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "No closure"
        ifil == 2 && (label = nothing)
        scatterlines!(
            nles,
            e_nm[1:ntrain, ifil];
            color = Cycled(1),
            linestyle,
            marker = :circle,
            label,
        )
    end
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "Smagorinsky"
        ifil == 2 && (label = nothing)
        scatterlines!(
            nles,
            e_smag[1:ntrain, ifil, iorder];
            color = Cycled(2),
            linestyle,
            marker = :utriangle,
            label,
        )
    end
    for ifil = 1:2
        linestyle = ifil == 1 ? :solid : :dash
        label = "CNN (prior)"
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
        linestyle = ifil == 1 ? :solid : :dash
        label = "CNN (post)"
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
    axislegend(; position = :lb)
    ylims!(ax, (T(0.025), T(1.00)))
    name = "$plotdir/convergence"
    ispath(name) || mkpath(name)
    save("$name/$(mname)_iorder$iorder.pdf", fig)
    fig
end

# Energy evolution ###########################################################
#
# Compute total kinetic energy as a function of time.

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
        psolver = psolver_spectral(setup)
        t = data_test.t
        ustart = data_test.data[ig, ifil].u[1] |> device
        tlims = (t[1], t[end])
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(ustart[1])
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
                solve_unsteady(; setup, ustart, tlims, Δt, processors, psolver)[2].ewriter
        end
        ke_smag[ig, ifil, iorder] =
            solve_unsteady(;
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = smagorinsky_closure(setup),
                ),
                ustart,
                tlims,
                Δt,
                processors,
                psolver,
                θ = θ_smag[ifil, iorder],
            )[2].ewriter
        ke_cnn_prior[ig, ifil, iorder] =
            solve_unsteady(;
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                ustart,
                tlims,
                Δt,
                processors,
                psolver,
                θ = θ_cnn_prior[ig, ifil],
            )[2].ewriter
        ke_cnn_post[ig, ifil, iorder] =
            solve_unsteady(;
                (;
                    setup...,
                    projectorder = getorder(iorder),
                    closure_model = wrappedclosure(closure, setup),
                ),
                ustart,
                tlims,
                Δt,
                processors,
                psolver,
                θ = θ_cnn_post[ig, ifil, iorder],
            )[2].ewriter
    end
    (; ke_ref, ke_nomodel, ke_smag, ke_cnn_prior, ke_cnn_post)
end;
clean();

# Plot energy evolution ########################################################

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    t = data_test.t
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            xlabel = "t",
            ylabel = "E(t)",
            title = "Kinetic energy: $lesmodel, $fil",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_ref[igrid, ifil];
            color = Cycled(1),
            linestyle = :dash,
            label = "Reference",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_nomodel[igrid, ifil];
            color = Cycled(1),
            label = "No closure",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_smag[igrid, ifil, iorder];
            color = Cycled(2),
            label = "Smagorinsky",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_cnn_prior[igrid, ifil, iorder];
            color = Cycled(3),
            label = "CNN (prior)",
        )
        lines!(
            ax,
            t,
            kineticenergy.ke_cnn_post[igrid, ifil, iorder];
            color = Cycled(4),
            label = "CNN (post)",
        )
        iorder == 1 && axislegend(; position = :lt)
        iorder == 2 && axislegend(; position = :lb)
        name = "$plotdir/energy_evolution/$mname/"
        ispath(name) || mkpath(name)
        save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
    end
end

# Compute Divergence ##########################################################
# 
# Compute divergence as a function of time.

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
        psolver = psolver_spectral(setup)
        t = data_test.t
        ustart = data_test.data[ig, ifil].u[1] |> device
        tlims = (t[1], t[end])
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(ustart[1])
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
        s(closure_model, θ) =
            solve_unsteady(;
                (; setup..., closure_model),
                ustart,
                tlims,
                method = RKProject(RK44(; T), getorder(iorder)),
                Δt,
                processors = (; dwriter),
                psolver,
                θ,
            )[2].dwriter
        iorder_use = iorder == 3 ? 2 : iorder
        d_nomodel[ig, ifil, iorder] = s(nothing, nothing)
        d_smag[ig, ifil, iorder] =
            s(smagorinsky_closure(setup), θ_smag[ifil, iorder_use])
        d_cnn_prior[ig, ifil, iorder] =
            s(wrappedclosure(closure, setup), θ_cnn_prior[ig, ifil])
        d_cnn_post[ig, ifil, iorder] =
            s(wrappedclosure(closure, setup), θ_cnn_post[ig, ifil, iorder_use])
    end
    (; d_ref, d_nomodel, d_smag, d_cnn_prior, d_cnn_post)
end;
clean();

# Check that divergence is within reasonable bounds
divs.d_ref .|> extrema
divs.d_nomodel .|> extrema
divs.d_smag .|> extrema
divs.d_cnn_prior .|> extrema
divs.d_cnn_post .|> extrema

# Plot Divergence #############################################################

# Better for PDF export
CairoMakie.activate!()

with_theme(;
    # fontsize = 20,
    palette,
) do
    t = data_test.t
    # for islog in (true, false)
    for islog in (false,)
        for iorder = 1:2, ifil = 1:2, igrid = 1:3
            println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
            lesmodel = if iorder == 1
                "DIF"
            elseif iorder == 2
                "DCF"
            elseif iorder == 3
                "DCF-RHS"
            end
            fil = ifil == 1 ? "FA" : "VA"
            nles = params_test.nles[igrid]
            fig = Figure(; size = (500, 400))
            ax = Axis(
                fig[1, 1];
                yscale = islog ? log10 : identity,
                xlabel = "t",
                title = "Divergence: $lesmodel, $fil,  $nles",
            )
            lines!(ax, t, divs.d_nomodel[igrid, ifil, iorder]; label = "No closure")
            lines!(ax, t, divs.d_smag[igrid, ifil, iorder]; label = "Smagorinsky")
            lines!(ax, t, divs.d_cnn_prior[igrid, ifil, iorder]; label = "CNN (prior)")
            lines!(ax, t, divs.d_cnn_post[igrid, ifil, iorder]; label = "CNN (post)")
            lines!(
                ax,
                t,
                divs.d_ref[igrid, ifil];
                color = Cycled(1),
                linestyle = :dash,
                label = "Reference",
            )
            iorder == 2 && ifil == 1 && axislegend(; position = :rt)
            islog && ylims!(ax, (T(1e-6), T(1e3)))
            name = "$plotdir/divergence/$mname/$(islog ? "log" : "lin")"
            ispath(name) || mkpath(name)
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
        psolver = psolver_spectral(setup)
        ustart = data_test.data[igrid, ifil].u[1] |> device
        tlims = (t[1], t[end])
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(ustart[1])
        s(closure_model, θ) =
            solve_unsteady(;
                (; setup..., closure_model),
                ustart,
                tlims,
                method = RKProject(RK44(; T), getorder(iorder)),
                Δt,
                psolver,
                θ,
            )[1].u .|> Array
        if iorder == 1
            # Does not depend on projection order
            u_ref[igrid, ifil] = data_test.data[igrid, ifil].u[end]
            u_nomodel[igrid, ifil] = s(nothing, nothing)
        end
        u_smag[igrid, ifil, iorder] =
            s(smagorinsky_closure(setup), θ_smag[ifil, iorder])
        u_cnn_prior[igrid, ifil, iorder] =
            s(wrappedclosure(closure, setup), θ_cnn_prior[igrid, ifil])
        u_cnn_post[igrid, ifil, iorder] =
            s(wrappedclosure(closure, setup), θ_cnn_post[igrid, ifil, iorder])
    end
    (; u_ref, u_nomodel, u_smag, u_cnn_prior, u_cnn_post)
end;
clean();

# # Save solution
# jldsave("$savepath/ufinal.jld2"; ufinal)
#
# # Load solution
# ufinal = load("$savepath/ufinal.jld2")["ufinal"];

# Plot spectra ###############################################################
# 
# Plot kinetic energy spectra at final time.

# Better for PDF export
CairoMakie.activate!()

fig = with_theme(; palette) do
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        lesmodel = iorder == 1 ? "DIF" : "DCF"
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
            up = u
            e = sum(up) do u
                u = u[Ip]
                uhat = fft(u)[ntuple(α -> 1:K[α], 2)...]
                abs2.(uhat) ./ (2 * prod(size(u))^2)
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
            xlabel = "κ",
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
        axislegend(ax; position = :cb)
        autolimits!(ax)
        ylims!(ax, (T(1e-3), T(0.35)))
        name = "$plotdir/energy_spectra/$mname"
        ispath(name) || mkpath(name)
        save("$(name)/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid).pdf", fig)
    end
end
clean();

# Plot fields ################################################################

# Export to PNG, otherwise each volume gets represented
# as a separate rectangle in the PDF
# (takes time to load in the article PDF)
GLMakie.activate!()

with_theme(; fontsize = 25, palette) do
    # Reference box for eddy comparison
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
    ]
    path = "$plotdir/les_fields/$mname"
    ispath(path) || mkpath(path)
    for iorder = 1:2, ifil = 1:2, igrid = 1:3
        setup = setups_test[igrid]
        name = "$path/iorder$(iorder)_ifilter$(ifil)_igrid$(igrid)"
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params_test.nles[igrid]
        function makeplot(u, title, suffix)
            fig = fieldplot(
                (; u, t = T(0));
                setup,
                title,
                docolorbar = false,
                size = (500, 500),
            )
            lines!(box; linewidth = 5, color = Cycled(2)) # Red in palette
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
