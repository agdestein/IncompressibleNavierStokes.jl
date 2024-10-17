# # A-posteriori analysis: Large Eddy Simulation (2D)
#
# This script is used to generate results for the the paper [Agdestein2024](@citet).
#
# - Generate filtered DNS data
# - Train closure models
# - Compare filters, closure models, and projection orders
#
# The filtered DNS data is saved and can be loaded in a subesequent session.
# The learned CNN parameters are also saved.

@info "Script started"

# Color palette for consistent theme throughout paper
palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])

# Choose where to put output
outdir = joinpath(@__DIR__, "output", "kolmogorov")
plotdir = joinpath(outdir, "plots")
logdir = joinpath(outdir, "logs")
ispath(outdir) || mkpath(outdir)
ispath(plotdir) || mkpath(plotdir)
ispath(logdir) || mkpath(logdir)

########################################################################## #src

# ## Configure logger

using PaperDC
using Dates

# Write output to file, as the default SLURM file is not updated often enough
logfile = joinpath(logdir, "log_$(Dates.now()).out")
# jobid = ENV["SLURM_JOB_ID"]
# taskid = ENV["SLURM_ARRAY_TASK_ID"]
# logfile = joinpath(logdir, "job=$(jobid)_task=$(taskid).out")
setsnelliuslogger(logfile)

@info "# A-posteriori analysis: Forced turbulence (2D)"

# ## Load packages

if false                      #src
    include("src/PaperDC.jl") #src
end                           #src

@info "Loading packages"

using Accessors
using Adapt
# using GLMakie
using CairoMakie
using CUDA
using IncompressibleNavierStokes
using IncompressibleNavierStokes.RKMethods
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using LuxCUDA
using NeuralClosure
using NNlib
using Optimisers
using Random
using SparseArrays
using FFTW

########################################################################## #src

# ## Random number seeds
#
# Use a new RNG with deterministic seed for each code "section"
# so that e.g. training batch selection does not depend on whether we
# generated fresh filtered DNS data or loaded existing one (the
# generation of which would change the state of a global RNG).
#
# Note: Using `rng = Random.default_rng()` twice seems to point to the
# same RNG, and mutating one also mutates the other.
# `rng = Xoshiro()` creates an independent copy each time.
#
# We define all the seeds here.

seeds = (;
    dns = 123, # Initial conditions
    θ₀ = 234, # Initial CNN parameters
    prior = 345, # A-priori training batch selection
    post = 456, # A-posteriori training batch selection
)

########################################################################## #src

# ## Hardware selection

# Precision
T = Float32

# Device
if CUDA.functional()
    ## For running on a CUDA compatible GPU
    @info "Running on CUDA"
    ArrayType = CuArray
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)
    clean() = (GC.gc(); CUDA.reclaim())
else
    ## For running on CPU.
    ## Consider reducing the sizes of DNS, LES, and CNN layers if
    ## you want to test run on a laptop.
    @warn "Running on CPU"
    ArrayType = Array
    device = identity
    clean() = nothing
end

########################################################################## #src

# ## Data generation
#
# Create filtered DNS data for training, validation, and testing.

# Parameters
params = (;
    D = 2,
    lims = (T(0), T(1)),
    Re = T(6e3),
    tburn = T(0.5),
    tsim = T(2),
    savefreq = 16,
    ndns = 2048,
    nles = [64, 128, 256],
    filters = (FaceAverage(), VolumeAverage()),
    ArrayType,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    method = RKMethods.Wray3(; T),
    bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y),
    issteadybodyforce = true,
    processors = (; log = timelogger(; nupdate = 100)),
)

# Data file names
ntrajectory = 10
dns_seeds = splitseed(seeds.dns, ntrajectory)
datadir = joinpath(outdir, "data")
ispath(datadir) || mkpath(datadir)
filenames =
    map(Iterators.product(params.nles, params.filters, dns_seeds)) do (nles, Φ, seed)
        "$datadir/seed=$(repr(seed))_filter=$(Φ)_nles=$(nles).jld2"
    end

create_data = false
create_data && for (iseed, seed) in enumerate(dns_seeds)
    @info "Creating DNS trajectory for seed $(repr(seed)) (DNS $iseed of $ntrajectory)"
    (; data, t, comptime) = create_les_data(; params..., rng = Xoshiro(seed))
    @info("Trajectory info:", comptime / 60, length(t), Base.summarysize(data) * 1e-9,)
    for ifilter in eachindex(params.filters), igrid in eachindex(params.nles)
        (; u, c) = data[igrid, ifilter]
        filename = filenames[igrid, ifilter, iseed]
        @info "Saving data to $filename"
        jldsave(filename; u, c, t, comptime)
    end
end

# Load filtered DNS data
data = namedtupleload.(filenames);
data_train = data[:, :, 1:8]
data_valid = data[:, :, 9:9]
data_test = data[:, :, 10:10]

# Computational time
sum(d -> d.comptime, data) / 60

# Build LES setup and assemble operators
setups = map(
    nles -> Setup(;
        x = ntuple(α -> LinRange(params.lims..., nles + 1), params.D),
        params.Re,
        params.ArrayType,
        params.bodyforce,
        params.issteadybodyforce,
    ),
    params.nles,
)

# Create input/output arrays for a-priori training (ubar vs c)
io_train = create_io_arrays(data_train, setups);
io_valid = create_io_arrays(data_valid, setups);
io_test = create_io_arrays(data_test, setups);

# Check that data is reasonably bounded
io_train[1].u |> extrema
io_train[1].c |> extrema
io_valid[1].u |> extrema
io_valid[1].c |> extrema
io_test[1].u |> extrema
io_test[1].c |> extrema

# Inspect data (live animation with GLMakie)
# GLMakie.activate!()
doplot = false
doplot && let
    ig = 2
    ifil = 1
    iseed = 1
    field, setup = data[ig, ifil, iseed].u, setups[ig]
    u = device.(field[1])
    o = Observable((; u, temp = nothing, t = nothing))
    ## energy_spectrum_plot(o; setup) |> display
    fig = fieldplot(
        o;
        setup,
        ## fieldname = :velocitynorm,
        ## fieldname = 1,
    )
    fig |> display
    for u in field[1:10:end]
        o[] = (; o[]..., u = device(u))
        fig |> display
        sleep(0.05)
    end
end

########################################################################## #src

# ## CNN closure model

# All training sessions will start from the same θ₀
# for a fair comparison.

closure, θ₀ = cnn(;
    setup = setups[1],
    radii = [2, 2, 2, 2, 2],
    channels = [24, 24, 24, 24, params.D],
    activations = [tanh, tanh, tanh, tanh, identity],
    use_bias = [true, true, true, true, false],
    rng = Xoshiro(seeds.θ₀),
);
closure.chain

@info "Initialized CNN with $(length(θ₀)) parameters"

# Give the CNN a test run
# Note: Data and parameters are stored on the CPU, and
# must be moved to the GPU before use (with `gpu_device`)
let
    @info "CNN warm up run"
    using NeuralClosure.Zygote
    u = io_train[1, 1].u |> x -> selectdim(x, ndims(x), 1:10) |> collect |> gpu_device()
    θ = θ₀ |> gpu_device()
    closure(u, θ)
    gradient(θ -> sum(closure(u, θ)), θ)
    clean()
end

########################################################################## #src

# ## Training

# ### A-priori training
#
# Train one set of CNN parameters for each of the filter types and grid sizes.
# Use the same batch selection random seed for each training setup.
# Save parameters to disk after each run.
# Plot training progress (for a validation data batch).

# Parameter save files
priorfiles = map(
    splat((nles, Φ) -> "$outdir/prior_filter=$(Φ)_nles=$(nles).jld2"),
    Iterators.product(params.nles, params.filters),
)

# Train
trainprior = false
for (ifil, Φ) in enumerate(params.filters), (ig, nles) in enumerate(params.nles)
    trainprior || break
    clean()
    starttime = time()
    @info "Training a-priori" Φ nles
    filename = priorfiles[ig, ifil]
    figname = joinpath(plotdir, splitext(basename(filename))[1] * ".pdf")
    checkpointname = join(splitext(filename), "_checkpoint")
    trainseed, validseed = splitseed(seeds.prior, 2) # Same seed for all training setups
    dataloader = create_dataloader_prior(io_train[ig, ifil]; batchsize = 50, device)
    θ = device(θ₀)
    loss = create_loss_prior(mean_squared_error, closure)
    opt = Adam(T(1.0e-3))
    optstate = Optimisers.setup(opt, θ)
    it = rand(Xoshiro(validseed), 1:size(io_valid[ig, ifil].u, params.D + 2), 50)
    validset =
        gpu_device()(map(v -> collect(selectdim(v, ndims(v), it)), io_valid[ig, ifil]))
    (; callbackstate, callback) = create_callback(
        create_relerr_prior(closure, validset...);
        θ,
        displayref = true,
        displayupdates = true, # Set to `true` if using CairoMakie
        figname,
        nupdate = 20,
    )
    trainstate = (; optstate, θ, rng = Xoshiro(trainseed))
    ncheck = 0
    if false
        # Resume from checkpoint
        (; ncheck, trainstate, callbackstate) = namedtupleload(checkpointname)
        trainstate = trainstate |> gpu_device()
        @reset callbackstate.θmin = callbackstate.θmin |> gpu_device()
    end
    for icheck = ncheck+1:10
        (; trainstate, callbackstate) =
            train(; dataloader, loss, trainstate, callbackstate, callback, niter = 1_000)
        # Save all states to resume training later
        # First move all arrays to CPU
        c = callbackstate |> cpu_device()
        t = trainstate |> cpu_device()
        jldsave(checkpointname; ncheck = icheck, callbackstate = c, trainstate = t)
    end
    θ = callbackstate.θmin # Use best θ instead of last θ
    prior = (; θ = Array(θ), comptime = time() - starttime, callbackstate.hist)
    save_object(filename, prior)
end
clean()

# Load learned parameters and training times
prior = load_object.(priorfiles)
θ_cnn_prior = map(p -> copyto!(copy(θ₀), p.θ), prior)

# Check that parameters are within reasonable bounds
θ_cnn_prior .|> extrema

# Training times
map(p -> p.comptime, prior)
map(p -> p.comptime, prior) |> vec
map(p -> p.comptime, prior) |> sum # Seconds
map(p -> p.comptime, prior) |> sum |> x -> x / 60 # Minutes
map(p -> p.comptime, prior) |> sum |> x -> x / 3600 # Hours

########################################################################## #src

# ### A-posteriori training
#
# Train one set of CNN parameters for each
# projection order, filter type and grid size.
# Use the same batch selection random seed for each training setup.
# Save parameters to disk after each combination.
# Plot training progress (for a validation data batch).
#
# The time stepper `RKProject` allows for choosing when to project.

projectorders = ProjectOrder.First, ProjectOrder.Last

# Parameter save files
postfiles = map(
    splat((nles, Φ, o) -> "$outdir/post_projectorder=$(o)_filter=$(Φ)_nles=$(nles).jld2"),
    Iterators.product(params.nles, params.filters, projectorders),
)

# Train
trainpost = false
for (iorder, projectorder) in enumerate(projectorders),
    (ifil, Φ) in enumerate(params.filters),
    (ig, nles) in enumerate(params.nles)

    trainpost || break
    clean()
    starttime = time()
    @info "Training a-posteriori" projectorder Φ nles

    filename = postfiles[ig, ifil, iorder]
    figname = joinpath(plotdir, splitext(basename(filename))[1] * ".pdf")
    checkpointname = join(splitext(filename), "_checkpoint")
    rng = Xoshiro(seeds.post) # Same seed for all training setups
    setup = setups[ig]
    psolver = psolver_spectral(setup)
    loss = create_loss_post(;
        setup,
        psolver,
        method = RKProject(params.method, projectorder),
        closure,
        nupdate = 2, # Time steps per loss evaluation
    )
    dataloader = create_dataloader_post(
        map(d -> (; d.u, d.t), data_train[ig, ifil, :]);
        device,
        nunroll = 20,
    )
    # θ = θ₀ |> gpu_device()
    θ = θ_cnn_prior[ig, ifil] |> gpu_device()
    opt = Adam(T(1.0e-3))
    optstate = Optimisers.setup(opt, θ)
    (; callbackstate, callback) = let
        d = data_valid[ig, ifil, 1]
        it = 1:50
        data = (; u = device.(d.u[it]), t = d.t[it])
        create_callback(
            create_relerr_post(;
                data,
                setup,
                psolver,
                method = RKProject(params.method, projectorder),
                closure_model = wrappedclosure(closure, setup),
                nupdate = 2,
            );
            θ,
            figname,
            displayref = false,
            displayupdates = true,
            nupdate = 10,
        )
    end
    trainstate = (; optstate, θ, rng = Xoshiro(seeds.post))
    ncheck = 0
    if false
        @info "Resuming from checkpoint $checkpointname"
        ncheck, trainstate, callbackstate = namedtupleload(checkpointname)
        trainstate = trainstate |> gpu_device()
        @reset callbackstate.θmin = callbackstate.θmin |> gpu_device()
    end
    for icheck = ncheck+1:10
        (; trainstate, callbackstate) =
            train(; dataloader, loss, trainstate, niter = 200, callbackstate, callback)
        @info "Saving checkpoint to $(basename(checkpointname))..."
        c = callbackstate |> cpu_device()
        t = trainstate |> cpu_device()
        jldsave(checkpointname; ncheck = icheck, callbackstate = c, trainstate = t)
        @info "... done"
    end
    θ = callbackstate.θmin # Use best θ instead of last θ
    results = (; θ = Array(θ), comptime = time() - starttime)
    save_object(filename, results)
end
clean()

# Load learned parameters and training times
post = load_object.(postfiles)
θ_cnn_post = map(p -> copyto!(copy(θ₀), p.θ), post)

# Check that parameters are within reasonable bounds
θ_cnn_post .|> extrema

# Training times
map(p -> p.comptime, post)
map(p -> p.comptime, post) |> x -> reshape(x, 6, 2)
map(p -> p.comptime, post) ./ 60
map(p -> p.comptime, post) |> sum
map(p -> p.comptime, post) |> sum |> x -> x / 60
map(p -> p.comptime, post) |> sum |> x -> x / 3600

########################################################################## #src

# ### Train Smagorinsky model
#
# Use a-posteriori error grid search to determine
# the optimal Smagorinsky constant.
# Find one constant for each projection order and filter type. but
# The constant is shared for all grid sizes, since the filter
# width (=grid size) is part of the model definition separately.

smagfiles = map(
    splat((Φ, o) -> "$outdir/smag_filter=$(Φ)_projectorder=$(o).jld2"),
    Iterators.product(params.filters, projectorders),
)

trainsmagorinsky = false
for (iorder, projectorder) in enumerate(projectorders),
    (ifil, Φ) in enumerate(params.filters)

    trainsmagorinsky || break
    clean()
    filename = smagfiles[ifil, iorder]
    starttime = time()
    θmin = T(0)
    emin = T(Inf)
    isample = 1
    it = 1:50
    for (iθ, θ) in enumerate(range(T(0), T(0.5), 501))
        iθ % 50 == 0 && @info "Testing Smagorinsky" projectorder Φ θ
        e = T(0)
        for (igrid, nles) in enumerate(params.nles)
            setup = setups[igrid]
            psolver = psolver_spectral(setup)
            d = data_train[igrid, ifil, isample]
            data = (; u = device.(d.u[it]), t = d.t[it])
            nupdate = 4
            err = create_relerr_post(;
                data,
                setup,
                psolver,
                method = RKProject(params.method, projectorder),
                closure_model = IncompressibleNavierStokes.smagorinsky_closure_natural(
                    setup,
                ),
                nupdate,
            )
            e += err(θ)
        end
        e /= length(params.nles)
        if e < emin
            emin = e
            θmin = θ
        end
    end
    results = (; θ = θmin, comptime = time() - starttime)
    save_object(filename, results)
end
clean()

# Load trained parameters
smag = load_object.(smagfiles)

# Extract coefficients
θ_smag = getfield.(smag, :θ)

# Computational time
getfield.(smag, :comptime)
getfield.(smag, :comptime) |> sum

########################################################################## #src

# ## Prediction errors

# ### Compute a-priori errors
#
# Note that it is still interesting to compute the a-priori errors for the
# a-posteriori trained CNN.
eprior = let
    prior = zeros(T, size(θ_cnn_prior))
    post = zeros(T, size(θ_cnn_post)...)
    for (ifil, Φ) in enumerate(params.filters), (ig, nles) in enumerate(params.nles)
        @info "Computing a-priori errors" Φ nles
        testset = io_test[ig, ifil]
        u, c = testset.u[:, :, :, 1:100], testset.c[:, :, :, 1:100]
        testset = (u, c) |> gpu_device()
        err = create_relerr_prior(closure, testset...)
        prior[ig, ifil] = err(gpu_device()(θ_cnn_prior[ig, ifil]))
        for iorder in eachindex(projectorders)
            post[ig, ifil, iorder] = err(gpu_device()(θ_cnn_post[ig, ifil, iorder]))
        end
    end
    (; prior, post)
end
clean()

io_test[1][1] |> size

eprior.prior
eprior.post

eprior.prior |> x -> reshape(x, :) |> x -> round.(x; digits = 2)
eprior.post |> x -> reshape(x, :, 2) |> x -> round.(x; digits = 2)

########################################################################## #src

# ### Compute a-posteriori errors

(; e_nm, e_smag, e_cnn, e_cnn_post) = let
    s = (length(params.nles), length(params.filters), length(projectorders))
    e_nm = zeros(T, s)
    e_smag = zeros(T, s)
    e_cnn = zeros(T, s)
    e_cnn_post = zeros(T, s)
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (ig, nles) in enumerate(params.nles)

        @info "Computing a-posteriori errors" projectorder Φ nles
        setup = setups[ig]
        psolver = psolver_spectral(setup)
        sample = data_test[ig, ifil, 1]
        it = 1:100
        data = (; u = device.(sample.u[it]), t = sample.t[it])
        nupdate = 16
        ## No model
        err =
            create_relerr_post(; data, setup, psolver, closure_model = nothing, nupdate)
        e_nm[ig, ifil, iorder] = err(nothing)
        ## Smagorinsky
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            method = RKProject(params.method, projectorder),
            closure_model = smagorinsky_closure(setup),
            nupdate,
        )
        e_smag[ig, ifil, iorder] = err(θ_smag[ifil, iorder])
        ## CNN
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            method = RKProject(params.method, projectorder),
            closure_model = wrappedclosure(closure, setup),
            nupdate,
        )
        e_cnn[ig, ifil, iorder] = err(gpu_device()(θ_cnn_prior[ig, ifil]))
        e_cnn_post[ig, ifil, iorder] = err(gpu_device()(θ_cnn_post[ig, ifil, iorder]))
    end
    (; e_nm, e_smag, e_cnn, e_cnn_post)
end
clean()

e_nm
e_smag
e_cnn
e_cnn_post

round.(
    [e_nm[:] reshape(e_smag, :, 2) reshape(e_cnn, :, 2) reshape(e_cnn_post, :, 2)][
        [1:3; 6:8],
        :,
    ];
    sigdigits = 2,
)

########################################################################## #src

# ### Plot a-priori errors

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    for (ifil, Φ) in enumerate(params.filters)
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            xscale = log10,
            xticks = params.nles,
            xlabel = "Resolution",
            title = "Relative a-priori error $(ifil == 1 ? " (FA)" : " (VA)")",
        )
        eprior_nm = ones(T, length(params.nles))
        for (e, marker, label, color) in [
            (eprior_nm, :circle, "No closure", Cycled(1)),
            (eprior.prior[:, ifil], :utriangle, "CNN (Lprior)", Cycled(2)),
            (eprior.post[:, ifil, 1], :rect, "CNN (Lpost, DIF)", Cycled(3)),
            (eprior.post[:, ifil, 2], :diamond, "CNN (Lpost, DCF)", Cycled(4)),
        ]
            scatterlines!(params.nles, e; marker, color, label)
        end
        axislegend(; position = :lb)
        ylims!(ax, (T(-0.05), T(1.05)))
        save("$plotdir/eprior_filter=$(Φ).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ### Plot a-posteriori errors

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    for (iorder, projectorder) in enumerate(projectorders)
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        nles = params.nles
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            xscale = log10,
            yscale = log10,
            xticks = nles,
            xlabel = "Resolution",
            title = "Relative error ($lesmodel)",
        )
        for (e, marker, label, color) in [
            (e_nm, :circle, "No closure", Cycled(1)),
            (e_smag, :utriangle, "Smagorinsky", Cycled(2)),
            (e_cnn, :rect, "CNN (Lprior)", Cycled(3)),
            (e_cnn_post, :diamond, "CNN (Lpost)", Cycled(4)),
        ]
            for ifil = 1:2
                linestyle = ifil == 1 ? :solid : :dash
                ifil == 2 && (label = nothing)
                scatterlines!(nles, e[:, ifil, iorder]; color, linestyle, marker, label)
            end
        end
        axislegend(; position = :rt)
        # ylims!(ax, (T(0.025), T(1.00)))
        save("$plotdir/epost_projectorder=$(projectorder).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ## Energy evolution

# ### Compute total kinetic energy as a function of time

kineticenergy = let
    clean()
    ngrid, nfilter, norder =
        length(params.nles), length(params.filters), length(projectorders)
    ke_ref = fill(zeros(Point2f, 0), ngrid, nfilter, norder)
    ke_nomodel = fill(zeros(Point2f, 0), ngrid, nfilter, norder)
    ke_smag = fill(zeros(Point2f, 0), ngrid, nfilter, norder)
    ke_cnn_prior = fill(zeros(Point2f, 0), ngrid, nfilter, norder)
    ke_cnn_post = fill(zeros(Point2f, 0), ngrid, nfilter, norder)
    for iorder = 1:norder, ifil = 1:nfilter, ig = 1:ngrid
        println("iorder = $iorder, ifil = $ifil, ig = $ig")
        projectorder = ProjectOrder.T(iorder)
        setup = setups[ig]
        psolver = psolver_spectral(setup)
        sample = data_test[ig, ifil, 1]
        ustart = sample.u[1] |> device
        tlims = (sample.t[1], sample.t[end])
        T = eltype(ustart[1])
        nupdate = 2
        ewriter = processor() do state
            ehist = zeros(Point2f, 0)
            on(state) do (; u, t, n)
                if n % nupdate == 0
                    e = total_kinetic_energy(u, setup)
                    push!(ehist, Point2f(t, e))
                end
            end
            state[] = state[] # Compute initial energy
            ehist
        end
        processors = (; ewriter)
        ## Does not depend on projection order
        ke_ref[ig, ifil, iorder] = map(
            (t, u) -> Point2f(t, total_kinetic_energy(device(u), setup)),
            sample.t,
            sample.u,
        )
        ke_nomodel[ig, ifil, iorder] =
            solve_unsteady(; setup, ustart, tlims, processors, psolver)[2].ewriter
        ke_smag[ig, ifil, iorder] =
            solve_unsteady(;
                setup = (;
                    setup...,
                    projectorder,
                    closure_model = smagorinsky_closure(setup),
                ),
                ustart,
                tlims,
                processors,
                psolver,
                θ = θ_smag[ifil, iorder],
            )[2].ewriter
        ke_cnn_prior[ig, ifil, iorder] =
            solve_unsteady(;
                setup = (;
                    setup...,
                    projectorder,
                    closure_model = wrappedclosure(closure, setup),
                ),
                ustart,
                tlims,
                processors,
                psolver,
                θ = gpu_device()(θ_cnn_prior[ig, ifil]),
            )[2].ewriter
        ke_cnn_post[ig, ifil, iorder] =
            solve_unsteady(;
                setup = (;
                    setup...,
                    projectorder,
                    closure_model = wrappedclosure(closure, setup),
                ),
                ustart,
                tlims,
                processors,
                psolver,
                θ = gpu_device()(θ_cnn_post[ig, ifil, iorder]),
            )[2].ewriter
    end
    (; ke_ref, ke_nomodel, ke_smag, ke_cnn_prior, ke_cnn_post)
end;
clean();

########################################################################## #src

# ### Plot energy evolution

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
        projectorder = ProjectOrder.T(iorder)
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            xlabel = "t",
            ylabel = "E(t)",
            title = "Kinetic energy: $lesmodel, $fil",
        )
        lines!(
            ax,
            kineticenergy.ke_ref[igrid, ifil, iorder];
            color = Cycled(1),
            linestyle = :dash,
            label = "Reference",
        )
        lines!(
            ax,
            kineticenergy.ke_nomodel[igrid, ifil, iorder];
            color = Cycled(1),
            label = "No closure",
        )
        lines!(
            ax,
            kineticenergy.ke_smag[igrid, ifil, iorder];
            color = Cycled(2),
            label = "Smagorinsky",
        )
        lines!(
            ax,
            kineticenergy.ke_cnn_prior[igrid, ifil, iorder];
            color = Cycled(3),
            label = "CNN (prior)",
        )
        lines!(
            ax,
            kineticenergy.ke_cnn_post[igrid, ifil, iorder];
            color = Cycled(4),
            label = "CNN (post)",
        )
        axislegend(; position = :lt)
        name = "$plotdir/energy_evolution/"
        ispath(name) || mkpath(name)
        save("$(name)/projectorder=$(projectorder)_filter=$(Φ)_nles=$(nles).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ## Divergence evolution

# ### Compute divergence as a function of time

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
        projectorder = ProjectOrder.T(iorder)
        setup = setups[ig]
        psolver = psolver_spectral(setup)
        t = data_test.t
        ustart = data_test.data[ig, ifil].u[1] |> device
        tlims = (t[1], t[end])
        nupdate = 2
        Δt = (t[2] - t[1]) / nupdate
        T = eltype(ustart[1])
        dwriter = processor() do state
            div = scalarfield(setup)
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
            ## Does not depend on projection order
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
                method = RKProject(RK44(; T), projectorder),
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

########################################################################## #src

# ### Plot Divergence

# Better for PDF export
CairoMakie.activate!()

with_theme(;
    ## fontsize = 20,
    palette,
) do
    t = data_test.t
    # for islog in (true, false)
    for islog in (false,)
        for iorder = 1:2, ifil = 1:2, igrid = 1:3
            println("iorder = $iorder, ifil = $ifil, igrid = $igrid")
            projectorder = ProjectOrder.T(iorder)
            lesmodel = if iorder == 1
                "DIF"
            elseif iorder == 2
                "DCF"
            elseif iorder == 3
                "DCF-RHS"
            end
            fil = ifil == 1 ? "FA" : "VA"
            nles = params.nles[igrid]
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

########################################################################## #src

# ## Solutions at final time

ufinal = let
    ngrid, nfilter, norder =
        length(params.nles), length(params.filters), length(projectorders)
    temp = ntuple(α -> zeros(T, 0, 0), 2)
    u_ref = fill(temp, ngrid, nfilter, norder)
    u_nomodel = fill(temp, ngrid, nfilter, norder)
    u_smag = fill(temp, ngrid, nfilter, norder)
    u_cnn_prior = fill(temp, ngrid, nfilter, norder)
    u_cnn_post = fill(temp, ngrid, nfilter, norder)
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        clean()
        @info "Computing test solutions" projectorder Φ nles
        setup = setups[igrid]
        psolver = psolver_spectral(setup)
        sample = data_test[igrid, ifil, 1]
        ustart = sample.u[1] |> gpu_device()
        t = sample.t
        tlims = (t[1], t[end])
        nupdate = 2
        T = eltype(ustart[1])
        s(closure_model, θ) =
            solve_unsteady(;
                setup = (; setup..., closure_model),
                ustart,
                tlims,
                method = RKProject(params.method, projectorder),
                psolver,
                θ,
            )[1].u .|> Array
        u_ref[igrid, ifil, iorder] = sample.u[end]
        u_nomodel[igrid, ifil, iorder] = s(nothing, nothing)
        u_smag[igrid, ifil, iorder] =
            s(smagorinsky_closure(setup), θ_smag[ifil, iorder])
        u_cnn_prior[igrid, ifil, iorder] =
            s(wrappedclosure(closure, setup), gpu_device()(θ_cnn_prior[igrid, ifil]))
        u_cnn_post[igrid, ifil, iorder] = s(
            wrappedclosure(closure, setup),
            gpu_device()(θ_cnn_post[igrid, ifil, iorder]),
        )
    end
    (; u_ref, u_nomodel, u_smag, u_cnn_prior, u_cnn_post)
end;
clean();

## # Save solution
## jldsave("$outdir/ufinal.jld2"; ufinal)
##
## # Load solution
## ufinal = load("$outdir/ufinal.jld2")["ufinal"];

########################################################################## #src

# ### Plot spectra
#
# Plot kinetic energy spectra at final time.

# Better for PDF export
CairoMakie.activate!()

fig = with_theme(; palette) do
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        @info "Plotting spectra" projectorder Φ nles
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params.nles[igrid]
        setup = setups[igrid]
        fields =
            [
                ufinal.u_ref[igrid, ifil, iorder],
                ufinal.u_nomodel[igrid, ifil, iorder],
                ufinal.u_smag[igrid, ifil, iorder],
                ufinal.u_cnn_prior[igrid, ifil, iorder],
                ufinal.u_cnn_post[igrid, ifil, iorder],
            ] .|> device
        (; Ip) = setup.grid
        (; inds, κ, K) = IncompressibleNavierStokes.spectral_stuff(setup)
        specs = map(fields) do u
            state = (; u)
            spec = observespectrum(state; setup)
            spec.ehat[]
        end
        kmax = maximum(κ)
        ## Build inertial slope above energy
        krange = [T(4), T(κ[end] / 2)]
        slope, slopelabel = -T(3), L"$\kappa^{-3}$"
        slopeconst = maximum(specs[1] ./ κ .^ slope)
        offset = 3
        inertia = offset .* slopeconst .* krange .^ slope
        ## Nice ticks
        logmax = round(Int, log2(kmax + 1))
        xticks = T(2) .^ (0:logmax)
        ## Make plot
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
        axislegend(ax; position = :lb)
        autolimits!(ax)
        # ylims!(ax, (T(1e-3), T(0.35)))
        name = "$plotdir/spectra/"
        ispath(name) || mkpath(name)
        save("$(name)/projectorder=$(projectorder)_filter=$(Φ)_nles=$(nles).pdf", fig)
        display(fig)
    end
end
clean();

########################################################################## #src

# ### Plot fields

# Export to PNG, otherwise each volume gets represented
# as a separate rectangle in the PDF
# (takes time to load in the article PDF)
GLMakie.activate!()

with_theme(; fontsize = 25, palette) do
    ## Reference box for eddy comparison
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
    path = "$plotdir/les_fields"
    ispath(path) || mkpath(path)
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        setup = setups[igrid]
        name = "$path/projectorder=$(projectorder)_filter=$(Φ)_nles=$(nles)"
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        nles = params.nles[igrid]
        function makeplot(u, title, suffix)
            fig = fieldplot(
                (; u, temp = nothing, t = T(0));
                setup,
                title,
                docolorbar = false,
                size = (500, 500),
            )
            lines!(box; linewidth = 5, color = Cycled(2)) # Red in palette
            fname = "$(name)_$(suffix).png"
            save(fname, fig)
            display(fig)
            ## run(`convert $fname -trim $fname`) # Requires imagemagick
        end
        makeplot(device(ufinal.u_ref[igrid, ifil, iorder]), "Reference, $fil, $nles", "ref")
        makeplot(
            device(ufinal.u_nomodel[igrid, ifil, iorder]),
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
