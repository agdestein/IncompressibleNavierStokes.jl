# LSP hack
if false
    include("src/SymmetryClosure.jl")
    include("../NeuralClosure/src/NeuralClosure.jl")
    include("../../src/IncompressibleNavierStokes.jl")
    using .SymmetryClosure
    using .NeuralClosure
    using .IncompressibleNavierStokes
end

# # A-posteriori analysis: Large Eddy Simulation (2D)
#
# Generate filtered DNS data, train closure model, compare filters,
# closure models, and projection orders.
#
# The filtered DNS data is saved and can be loaded in a subesequent session.
# The learned CNN parameters are also saved.

using Adapt
# using GLMakie
using CairoMakie
using IncompressibleNavierStokes
using JLD2
using LinearAlgebra
using NeuralClosure
using NNlib
using Optimisers
using Random
using SymmetryClosure

# Choose where to put output
plotdir = joinpath(@__DIR__, "output", "plots")
datadir = joinpath(@__DIR__, "output", "data")
ispath(plotdir) || mkpath(plotdir)
ispath(datadir) || mkpath(datadir)

# Random number generator seeds ################################################
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
# We define all the seeds here so that we don't accidentally type the same seed
# twice.

seeds = (;
    dns = 123, # Initial conditions
    θ₀ = 234, # Initial CNN parameters
    training = 345, # Training batch selection
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
rng = Xoshiro(seeds.dns)

# Parameters
get_params(nlesscalar) = (;
    D = 2,
    Re = T(10_000),
    tburn = T(0.05),
    tsim = T(0.5),
    Δt = T(5e-5),
    nles = map(n -> (n, n), nlesscalar), # LES resolutions
    ndns = (n -> (n, n))(4096), # DNS resolution
    # ndns = (n -> (n, n))(1024), # DNS resolution
    filters = (FaceAverage(),),
    ArrayType,
    create_psolver = psolver_spectral,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    rng,
)

# Get parameters for multiple LES resolutions
nles = [64, 128, 256]
params_train = (; get_params(nles)..., tsim = T(0.5), savefreq = 10);
params_valid = (; get_params(nles)..., tsim = T(0.1), savefreq = 40);
params_test = (; get_params(nles)..., tsim = T(0.1), savefreq = 10);

create_data = false
if create_data
    # Create filtered DNS data
    data_train = [create_les_data(; params_train...) for _ = 1:5]
    data_valid = [create_les_data(; params_valid...) for _ = 1:1]
    data_test = [create_les_data(; params_test...) for _ = 1:1]

    # Save filtered DNS data
    jldsave("$datadir/data_train.jld2"; data_train)
    jldsave("$datadir/data_valid.jld2"; data_valid)
    jldsave("$datadir/data_test.jld2"; data_test)
end

# Load filtered DNS data
data_train = load("$datadir/data_train.jld2", "data_train");
data_valid = load("$datadir/data_valid.jld2", "data_valid");
data_test = load("$datadir/data_test.jld2", "data_test");

# Computational time
data_train[1].comptime
data_valid[1].comptime
data_test[1].comptime
map(d -> d.comptime, data_train)
sum(d -> d.comptime, data_train) / 60
data_test[1].comptime / 60
sum(dd -> sum(d -> d.comptime, dd), (data_train, data_valid, data_test))

# Build LES setup and assemble operators
getsetups(params) =
    map(params.nles) do nles
        x = ntuple(α -> LinRange(T(0), T(1), nles[α] + 1), params.D)
        Setup(x...; params.Re, params.ArrayType)
    end

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
io_test = create_io_arrays(data_test, setups_test);

# Check that data is reasonably bounded
io_train[1].u |> extrema
io_train[1].c |> extrema
io_valid[1].u |> extrema
io_valid[1].c |> extrema
io_test[1].u |> extrema
io_test[1].c |> extrema

# Inspect data (live animation with GLMakie)
# GLMakie.activate!()
let
    ig = 2
    field, setup = data_train[1].data[ig].u, setups_train[ig]
    # field, setup = data_valid[1].data[ig].u, setups_valid[ig];
    # field, setup = data_test[.data[ig].u, setups_test[ig];
    u = device.(field[1])
    o = Observable((; u, temp = nothing, t = nothing))
    # energy_spectrum_plot(o; setup) |> display
    fig = fieldplot(
        o;
        setup,
        # fieldname = :velocitynorm,
        # fieldname = 1,
    )
    fig |> display
    for i in eachindex(field)
        i % 50 == 0 || continue
        o[] = (; o[]..., u = device(field[i]))
        fig |> display
        sleep(0.1)
    end
end

# CNN closure models #########################################################

# Random number generator for initial CNN parameters.
# All training sessions will start from the same θ₀
# for a fair comparison.

# Regular CNN
m_cnn = let
    rng = Xoshiro(seeds.θ₀)
    name = "cnn"
    closure, θ₀ = cnn(;
        setup = setups_train[1],
        radii = [2, 2, 2, 2, 2],
        channels = [24, 24, 24, 24, params_train.D],
        activations = [tanh, tanh, tanh, tanh, identity],
        use_bias = [true, true, true, true, false],
        rng,
    )
    (; closure, θ₀, name)
end;
m_cnn.closure.chain

# Group CNN: Same number of channels as regular CNN
m_gcnn_a = let
    rng = Xoshiro(seeds.θ₀)
    name = "gcnn_a"
    closure, θ₀ = gcnn(;
        setup = setups_train[1],
        radii = [2, 2, 2, 2, 2],
        channels = [6, 6, 6, 6, 1],
        activations = [tanh, tanh, tanh, tanh, identity],
        use_bias = [true, true, true, true, false],
        rng,
    )
    (; closure, θ₀, name)
end;
m_gcnn_a.closure.chain

# Group CNN: Same number of parameters as regular CNN
m_gcnn_b = let
    rng = Xoshiro(seeds.θ₀)
    name = "gcnn_b"
    closure, θ₀ = gcnn(;
        setup = setups_train[1],
        radii = [2, 2, 2, 2, 2],
        channels = [12, 12, 12, 12, 1],
        activations = [tanh, tanh, tanh, tanh, identity],
        use_bias = [true, true, true, true, false],
        rng,
    )
    (; closure, θ₀, name)
end;
m_gcnn_b.closure.chain

models = m_cnn, m_gcnn_a, m_gcnn_b;

# Give the CNN a test run
# Note: Data and parameters are stored on the CPU, and
# must be moved to the GPU before running (`device`)
models[1].closure(device(io_train[1].u[:, :, :, 1:50]), device(models[1].θ₀));
models[2].closure(device(io_train[1].u[:, :, :, 1:50]), device(models[2].θ₀));
models[3].closure(device(io_train[1].u[:, :, :, 1:50]), device(models[3].θ₀));

# A-priori training ###########################################################
#
# Train one set of CNN parameters for each of the filter types and grid sizes.
# Save parameters to disk after each run.
# Plot training progress (for a validation data batch).

priorfiles = broadcast(eachindex(nles), eachindex(models)') do ig, im
    m = models[im]
    "$datadir/prior_$(m.name)_igrid$ig.jld2"
end

# Train
dotrain = false
dotrain && let
    # Random number generator for batch selection
    rng = Xoshiro(seeds.training)
    for (im, m) in enumerate(models), ig = 1:length(nles)
        plotfile = "$plotdir/training_prior_$(m.name)_igrid$ig.pdf"
        clean()
        starttime = time()
        @info "Training for $(m.name), grid $ig"
        d = create_dataloader_prior(io_train[ig]; batchsize = 100, device, rng)
        θ = device(m.θ₀)
        loss = create_loss_prior(mean_squared_error, m.closure)
        opt = Optimisers.setup(Adam(T(1.0e-3)), θ)
        it = rand(rng, 1:size(io_valid[ig].u, 4), 50)
        validset = device(map(v -> v[:, :, :, it], io_valid[ig]))
        (; callbackstate, callback) = create_callback(
            create_relerr_prior(m.closure, validset...);
            θ,
            displayref = true,
            display_each_iteration = true, # Set to `true` if using CairoMakie
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
        jldsave(priorfiles[ig, im]; prior)
        save(plotfile, current_figure())
    end
    clean()
end

# Load learned parameters and training times
prior = load.(priorfiles, "prior");
θ_cnn_prior = broadcast(eachindex(nles), eachindex(models)') do ig, im
    m = models[im]
    p = prior[ig, im]
    copyto!(device(m.θ₀), p.θ)
end;

# Check that parameters are within reasonable bounds
using Statistics
CUDA.@allowscalar θ_cnn_prior[1] |> std
CUDA.@allowscalar θ_cnn_prior[2] |> std
CUDA.@allowscalar θ_cnn_prior[3] |> std
θ_cnn_prior[1] |> extrema
θ_cnn_prior[2] |> extrema
θ_cnn_prior[3] |> extrema

# Training times
map(p -> p.comptime, prior)
map(p -> p.comptime, prior) |> vec
map(p -> p.comptime, prior) |> sum # Seconds
map(p -> p.comptime, prior) |> sum |> x -> x / 60 # Minutes
map(p -> p.comptime, prior) |> sum |> x -> x / 3600 # Hours

# Compute a-priori errors ###################################################
#
# Note that it is still interesting to compute the a-priori errors for the
# a-posteriori trained CNN.

e_prior = let
    e = zeros(T, length(nles), length(models))
    for (im, m) in enumerate(models), ig = 1:length(nles)
        @info "Computing a-priori error for $(m.name), grid $ig"
        testset = device(io_test[ig])
        err = create_relerr_prior(m.closure, testset...)
        e[ig, im] = err(θ_cnn_prior[ig, im])
    end
    e
end
clean()

e_prior

# Compute a-posteriori errors #################################################

(; e_nm, e_m) = let
    e_nm = zeros(T, length(nles))
    e_m = zeros(T, length(nles), length(models))
    for ig = 1:size(data_test[1].data, 1)
        clean()
        setup = setups_test[ig]
        psolver = psolver_spectral(setup)
        data = (; u = device.(data_test[1].data[ig].u), t = data_test[1].t)
        nupdate = 2
        @info "Computing a-posteriori error for no-model, grid $ig"
        err =
            create_relerr_post(; data, setup, psolver, closure_model = nothing, nupdate)
        e_nm[ig] = err(nothing)
        # CNN
        for (im, m) in enumerate(models)
            @info "Computing a-posteriori error for $(m.name), grid $ig"
            err = create_relerr_post(;
                data,
                setup,
                psolver,
                closure_model = wrappedclosure(m.closure, setup),
                nupdate,
            )
            e_m[ig, im] = err(θ_cnn_prior[ig, im])
        end
    end
    (; e_nm, e_m)
end
clean()

e_nm
e_m

# Compute symmetry errors #####################################################

e_symm_prior = let
    e = zeros(T, length(nles), length(models))
    for (im, m) in enumerate(models), ig = 1:length(nles)
        @info "Computing a-priori equivariance error for $(m.name), grid $ig"
        setup = setups_test[ig]
        setup = (; setup..., closure_model = wrappedclosure(m.closure, setup))
        err =
            create_relerr_symmetry_prior(; u = device.(data_test[1].data[ig].u), setup)
        e[ig, im] = err(θ_cnn_prior[ig, im])
    end
    e
end
clean()

e_symm_prior

e_symm_post = let
    e = zeros(T, length(nles), length(models))
    for (im, m) in enumerate(models), ig = 1:size(data_test[1].data, 1)
        @info "Computing a-posteriori equivariance error for $(m.name), grid $ig"
        setup = setups_test[ig]
        setup = (; setup..., closure_model = wrappedclosure(m.closure, setup))
        err = create_relerr_symmetry_post(;
            u = device.(data_test[1].data[ig].u[1]),
            setup,
            psolver = psolver_spectral(setup),
            Δt = (data_test[1].t[2] - data_test[1].t[1]) / 2,
            nstep = 10, # length(data_test[1].t) - 1,
            g = 1,
        )
        e[ig, im] = err(θ_cnn_prior[ig, im])
    end
    e
end
clean()

e_symm_post

# Plot errors #################################################################

let
    for (e, title, filename) in [
        (e_prior, "A-priori error", "error_prior.pdf"),
        (e_m, "A-posteriori error", "error_post.pdf"),
        (e_symm_prior, "A-priori equivariance error", "error_symm_prior.pdf"),
        (e_symm_post, "A-posteriori equivariance error", "error_symm_post.pdf"),
    ]
        fig = Figure()
        ax = Axis(
            fig[1, 1];
            xscale = log10,
            yscale = log10,
            xticks = nles,
            xlabel = "LES grid size",
            ylabel = "Relative error",
            title,
        )
        markers = [:circle, :utriangle, :rect, :diamond]
        for (i, m) in enumerate(models)
            scatterlines!(ax, nles, e[:, i]; marker = markers[i], label = m.name)
        end
        axislegend(ax)
        display(fig)
        save(joinpath(plotdir, filename), fig)
    end
end
