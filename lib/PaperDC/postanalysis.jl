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

if false                      #src
    include("src/PaperDC.jl") #src
end                           #src

@info "Script started"

# Color palette for consistent theme throughout paper
palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])

# Choose where to put output
basedir = haskey(ENV, "DEEPDIP") ? ENV["DEEPDIP"] : @__DIR__
outdir = joinpath(basedir, "output", "kolmogorov")
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
isslurm = haskey(ENV, "SLURM_JOB_ID")
if isslurm
    jobid = parse(Int, ENV["SLURM_JOB_ID"])
    taskid = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    logfile = "job=$(jobid)_task=$(taskid)_$(Dates.now()).out"
else
    taskid = nothing
    logfile = "log_$(Dates.now()).out"
end
logfile = joinpath(logdir, logfile)
setsnelliuslogger(logfile)

@info "# A-posteriori analysis: Forced turbulence (2D)"

# ## Load packages

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
    θ_start = 234, # Initial CNN parameters
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
    tsim = T(5),
    savefreq = 50,
    ndns = 4096,
    nles = [32, 64, 128, 256],
    filters = (FaceAverage(), VolumeAverage()),
    ArrayType,
    icfunc = (setup, psolver, rng) -> random_field(setup, T(0); kp = 20, psolver, rng),
    method = RKMethods.Wray3(; T),
    bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y),
    issteadybodyforce = true,
    processors = (; log = timelogger(; nupdate = 100)),
)

# DNS seeds
ntrajectory = 8
dns_seeds = splitseed(seeds.dns, ntrajectory)
dns_seeds_train = dns_seeds[1:ntrajectory-2]
dns_seeds_valid = dns_seeds[ntrajectory-1:ntrajectory-1]
dns_seeds_test = dns_seeds[ntrajectory:ntrajectory]

# Create data
docreatedata = false
docreatedata && createdata(; params, seeds = dns_seeds, outdir, taskid)

# Computational time
docomp = false
docomp && let
    comptime, datasize = 0.0, 0.0
    for seed in dns_seeds
        comptime += load(
            getdatafile(outdir, params.nles[1], params.filters[1], seed),
            "comptime",
        )
    end
    for seed in dns_seeds, nles in params.nles, Φ in params.filters
        data = namedtupleload(getdatafile(outdir, nles, Φ, seed))
        datasize += Base.summarysize(data)
    end
    @info "Data" comptime / 60 datasize * 1e-9
    clean()
end

# LES setups
setups = map(nles -> getsetup(; params, nles), params.nles);

########################################################################## #src

# ## CNN closure model

# All training sessions will start from the same θ₀
# for a fair comparison.

closure, θ_start = cnn(;
    setup = setups[1],
    radii = [2, 2, 2, 2, 2],
    channels = [24, 24, 24, 24, params.D],
    activations = [tanh, tanh, tanh, tanh, identity],
    use_bias = [true, true, true, true, false],
    rng = Xoshiro(seeds.θ_start),
);
closure.chain

@info "Initialized CNN with $(length(θ_start)) parameters"

# Give the CNN a test run
# Note: Data and parameters are stored on the CPU, and
# must be moved to the GPU before use (with `device`)
let
    @info "CNN warm up run"
    using NeuralClosure.Zygote
    u = randn(T, 32, 32, 2, 10) |> device
    θ = θ_start |> device
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

# Train
dotrainprior = false
dotrainprior && trainprior(;
    params,
    priorseed = seeds.prior,
    dns_seeds_train,
    dns_seeds_valid,
    taskid,
    outdir,
    plotdir,
    closure,
    θ_start,
    opt = Adam(T(1.0e-3)),
    nvalid = 50,
    batchsize = 50,
    displayref = true,
    displayupdates = true, # Set to `true` if using CairoMakie
    nupdate_callback = 20,
    loadcheckpoint = false,
    ncheckpoint = 10,
    niter = 1000,
)

# Load learned parameters and training times
priortraining = loadprior(outdir, params.nles, params.filters)
θ_cnn_prior = map(p -> copyto!(copy(θ_start), p.θ), priortraining)
@info "" θ_cnn_prior .|> extrema # Check that parameters are within reasonable bounds

# Training times
map(p -> p.comptime, priortraining)
map(p -> p.comptime, priortraining) |> sum |> x -> x / 60 # Minutes

# ## Plot training history

with_theme(; palette) do
    fig = Figure(; size = (1000, 400))
    for (ifil, Φ) in enumerate(params.filters), (ig, nles) in enumerate(params.nles)
        fil = Φ isa FaceAverage ? "FA" : "VA"
        xlabel = ifil == 2 ? "Iteration" : ""
        ylabel = ig == 1 ? Φ isa FaceAverage ? "Error (FA)" : "Error (VA)" : ""
        title = ifil == 1 ? "n = $(nles)" : ""
        xticksvisible = ifil == 2
        xticklabelsvisible = ifil == 2
        yticksvisible = ig == 1
        yticklabelsvisible = ig == 1
        ax = Axis(
            fig[ifil, ig];
            title,
            xlabel,
            ylabel,
            xticksvisible,
            xticklabelsvisible,
            yticksvisible,
            yticklabelsvisible,
        )
        lines!(ax, priortraining[ig, ifil].hist)
    end
    linkaxes!(filter(x -> x isa Axis, fig.content)...)
    fig
end

with_theme(; palette) do
    for (ig, nles) in enumerate(params.nles)
        fig = Figure(; size = (350, 300))
        ax = Axis(
            fig[1, 1];
            title = "A-priori training error, n = $(nles)",
            xlabel = "Iteration",
        )
        for (ifil, Φ) in enumerate(params.filters)
            lines!(
                ax,
                priortraining[ig, ifil].hist;
                label = Φ isa FaceAverage ? "FA" : "VA",
            )
        end
        axislegend(ax)
        figdir = joinpath(plotdir, "priortraining")
        ispath(figdir) || mkpath(figdir)
        save("$figdir/nles=$(nles).pdf", fig)
        display(fig)
    end
end

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

# Train
dotrainpost = false
dotrainpost && trainpost(;
    params,
    projectorders,
    outdir,
    plotdir,
    taskid,
    postseed = seeds.post,
    dns_seeds_train,
    dns_seeds_valid,
    nsubstep = 5,
    nunroll = 20,
    closure,
    θ_start = θ_cnn_prior,
    opt = Adam(T(1.0e-3)),
    nunroll_valid = 50,
    nupdate_callback = 10,
    displayref = false,
    displayupdates = true,
    loadcheckpoint = false,
    ncheckpoint = 10,
    niter = 200,
)

# Load learned parameters and training times

posttraining = loadpost(outdir, params.nles, params.filters, projectorders)
θ_cnn_post = map(p -> copyto!(copy(θ_start), p.θ), posttraining)
@info "" θ_cnn_post .|> extrema # Check that parameters are within reasonable bounds

# Training times
map(p -> p.comptime, posttraining) ./ 60
map(p -> p.comptime, posttraining) |> sum |> x -> x / 60

########################################################################## #src

# ### Train Smagorinsky model
#
# Use a-posteriori error grid search to determine the optimal Smagorinsky constant.
# Find one constant for each projection order, filter type, and grid size.

dotrainsmagorinsky = false
dotrainsmagorinsky && trainsmagorinsky(;
    params,
    projectorders,
    outdir,
    dns_seeds_train,
    taskid,
    nunroll = 50,
    nsubstep = 5,
    ninfo = 50,
    θrange = range(t(0), t(0.3), 301),
)

# Load trained parameters
smag = loadsmagorinsky(outdir, params.nles, params.filters, projectorders)

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
    eprior = (;
        nomodel = ones(T, length(params.nles)),
        prior = zeros(T, size(θ_cnn_prior)),
        post = zeros(T, size(θ_cnn_post)),
    )
    for (ifil, Φ) in enumerate(params.filters), (ig, nles) in enumerate(params.nles)
        @info "Computing a-priori errors" Φ nles

        setup = setups[ig]
        data = map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_test)
        testset = create_io_arrays(data, setup)
        i = 1:100
        u, c = testset.u[:, :, :, i], testset.c[:, :, :, i]
        testset = (u, c) |> device
        err = create_relerr_prior(closure, testset...)
        eprior.prior[ig, ifil] = err(device(θ_cnn_prior[ig, ifil]))
        for iorder in eachindex(projectorders)
            eprior.post[ig, ifil, iorder] = err(device(θ_cnn_post[ig, ifil, iorder]))
        end
    end
    eprior
end
clean()

########################################################################## #src

# ### Compute a-posteriori errors

epost = let
    s = (length(params.nles), length(params.filters), length(projectorders))
    epost = (;
        nomodel = zeros(T, s),
        smag = zeros(T, s),
        cnn_prior = zeros(T, s),
        cnn_post = zeros(T, s),
    )
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (ig, nles) in enumerate(params.nles)

        @info "Computing a-posteriori errors" projectorder Φ nles
        setup = setups[ig]
        psolver = psolver_spectral(setup)
        sample = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_test[1]))
        it = 1:100
        data = (; u = device.(sample.u[it]), t = sample.t[it])
        nupdate = 5
        ## No model
        err =
            create_relerr_post(; data, setup, psolver, closure_model = nothing, nupdate)
        epost.nomodel[ig, ifil, iorder] = err(nothing)
        ## Smagorinsky
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            method = RKProject(params.method, projectorder),
            closure_model = smagorinsky_closure(setup),
            nupdate,
        )
        epost.smag[ig, ifil, iorder] = err(θ_smag[ig, ifil, iorder])
        ## CNN
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            method = RKProject(params.method, projectorder),
            closure_model = wrappedclosure(closure, setup),
            nupdate,
        )
        epost.cnn_prior[ig, ifil, iorder] = err(device(θ_cnn_prior[ig, ifil]))
        epost.cnn_post[ig, ifil, iorder] = err(device(θ_cnn_post[ig, ifil, iorder]))
        clean()
    end
    epost
end

epost.nomodel
epost.smag
epost.cnn_prior
epost.cnn_post

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
        for (e, marker, label, color) in [
            (eprior.nomodel, :circle, "No closure", Cycled(1)),
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
        (; nles) = params
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            xscale = log10,
            yscale = log10,
            xticks = nles,
            xlabel = "Resolution",
            title = "Relative a-posteriori error ($lesmodel)",
        )
        for (e, marker, label, color) in [
            (epost.nomodel, :circle, "No closure", Cycled(1)),
            (epost.smag, :utriangle, "Smagorinsky", Cycled(2)),
            (epost.cnn_prior, :rect, "CNN (Lprior)", Cycled(3)),
            (epost.cnn_post, :diamond, "CNN (Lpost)", Cycled(4)),
        ]
            for ifil = 1:2
                linestyle = ifil == 1 ? :solid : :dash
                ifil == 2 && (label = nothing)
                scatterlines!(nles, e[:, ifil, iorder]; color, linestyle, marker, label)
            end
        end
        axislegend(; position = :lb)
        # ylims!(ax, (T(0.025), T(1.00)))
        save("$plotdir/epost_projectorder=$(projectorder).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ## Energy evolution

# ### Compute total kinetic energy as a function of time

divergencehistory, energyhistory = let
    s = length(params.nles), length(params.filters), length(projectorders)
    keys = [:ref, :nomodel, :smag, :cnn_prior, :cnn_post]
    divergencehistory = (; map(k -> k => fill(Point2f[], s), keys)...)
    energyhistory = (; map(k -> k => fill(Point2f[], s), keys)...)
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (ig, nles) in enumerate(params.nles)

        I = CartesianIndex(ig, ifil, iorder)
        @info "Computing divergence and kinetic energy" projectorder Φ nles
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        sample = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_test[1]))
        ustart = sample.u[1] |> device
        T = eltype(ustart[1])

        # Reference trajectories
        divergencehistory.ref[ig, ifil, iorder] = let
            div = scalarfield(setup)
            udev = vectorfield(setup)
            map(sample.t[1:10:end], sample.u[1:10:end]) do t, u
                copyto!.(udev, u)
                IncompressibleNavierStokes.divergence!(div, udev, setup)
                d = view(div, setup.grid.Ip)
                d = sum(abs2, d) / length(d)
                d = sqrt(d)
                Point2f(t, d)
            end
        end
        energyhistory.ref[ig, ifil, iorder] = map(
            (t, u) -> Point2f(t, total_kinetic_energy(device(u), setup)),
            sample.t,
            sample.u,
        )

        nupdate = 5
        writer = processor() do state
            div = scalarfield(setup)
            dhist = Point2f[]
            ehist = zeros(Point2f, 0)
            on(state) do (; u, t, n)
                if n % nupdate == 0
                    IncompressibleNavierStokes.divergence!(div, u, setup)
                    d = view(div, setup.grid.Ip)
                    d = sum(abs2, d) / length(d)
                    d = sqrt(d)
                    push!(dhist, Point2f(t, d))
                    e = total_kinetic_energy(u, setup)
                    push!(ehist, Point2f(t, e))
                end
            end
            state[] = state[] # Compute initial divergence
            (; dhist, ehist)
        end

        for (sym, closure_model, θ) in [
            (:nomodel, nothing, nothing),
            (:smag, smagorinsky_closure(setup), θ_smag[I]),
            (:cnn_prior, wrappedclosure(closure, setup), device(θ_cnn_prior[ig, ifil])),
            (:cnn_post, wrappedclosure(closure, setup), device(θ_cnn_post[I])),
        ]
            _, results = solve_unsteady(;
                setup = (; setup..., closure_model),
                ustart,
                tlims = (sample.t[1], sample.t[end]),
                Δt_min = T(1e-5),
                method = RKProject(params.method, projectorder),
                processors = (; writer, logger = timelogger(; nupdate = 1000)),
                psolver,
                θ,
            )
            divergencehistory[sym][I] = results.writer.dhist
            energyhistory[sym][I] = results.writer.ehist
        end
    end
    divergencehistory, energyhistory
end;
clean();

########################################################################## #src

# Check that energy is within reasonable bounds
energyhistory.ref .|> extrema
energyhistory.nomodel .|> extrema
energyhistory.smag .|> extrema
energyhistory.cnn_prior .|> extrema
energyhistory.cnn_post .|> extrema

# Check that divergence is within reasonable bounds
divergencehistory.ref .|> extrema
divergencehistory.nomodel .|> extrema
divergencehistory.smag .|> extrema
divergencehistory.cnn_prior .|> extrema
divergencehistory.cnn_post .|> extrema

########################################################################## #src

# ### Plot energy evolution

# Better for PDF export
CairoMakie.activate!()

with_theme(; palette) do
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        @info "Plotting energy evolution" projectorder Φ nles
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            # xscale = log10,
            # yscale = log10,
            xlabel = "t",
            ylabel = "E(t)",
            title = "Kinetic energy: $lesmodel, $fil",
        )
        # xlims!(ax, (1e-2, 5.0))
        # xlims!(ax, (0.0, 1.0))
        # ylims!(ax, (1.3, 2.3))
        plots = [
            (energyhistory.ref, :dash, 1, "Reference"),
            (energyhistory.nomodel, :solid, 1, "No closure"),
            (energyhistory.smag, :solid, 2, "Smagorinsky"),
            (energyhistory.cnn_prior, :solid, 3, "CNN (prior)"),
            (energyhistory.cnn_post, :solid, 4, "CNN (post)"),
        ]
        for (p, linestyle, i, label) in plots
            lines!(ax, p[igrid, ifil, iorder]; color = Cycled(i), linestyle, label)
        end
        axislegend(; position = :lt)

        # # Plot zoom-in box
        # tlims = 1.0, 1.7
        # klims = 1.7, 2.1
        # box = [
        #     Point2f(tlims[1], klims[1]),
        #     Point2f(tlims[2], klims[1]),
        #     Point2f(tlims[2], klims[2]),
        #     Point2f(tlims[1], klims[2]),
        #     Point2f(tlims[1], klims[1]),
        # ]
        # lines!(ax, box; color = :black)
        # ax2 = Axis(
        #     fig[1,1];
        #     # bbox = BBox(0.8, 0.9, 0.2, 0.3),
        #     width=Relative(0.3),
        #     height=Relative(0.3),
        #     halign=0.95,
        #     valign=0.05,
        #     limits = (tlims..., klims...),
        #     xscale = log10,
        #     yscale = log10,
        #     xticksvisible = false,
        #     xticklabelsvisible = false,
        #     xgridvisible = false,
        #     yticksvisible = false,
        #     yticklabelsvisible = false,
        #     ygridvisible = false,
        #     backgroundcolor = :white,
        # )
        # # https://discourse.julialang.org/t/makie-inset-axes-and-their-drawing-order/60987/5
        # translate!(ax2.scene, 0, 0, 10)
        # translate!(ax2.elements[:background], 0, 0, 9)
        # for (sym, linestyle, i, label) in plots
        #     lines!(
        #         ax2,
        #         getfield(kineticenergy, sym)[igrid, ifil, iorder];
        #         color = Cycled(i),
        #         linestyle,
        #     )
        # end

        name = "$plotdir/energy_evolution/"
        ispath(name) || mkpath(name)
        save("$(name)/projectorder=$(projectorder)_filter=$(Φ)_nles=$(nles).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ### Plot Divergence

# Better for PDF export
CairoMakie.activate!()

with_theme(;
    ## fontsize = 20,
    palette,
) do
    islog = true
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        @info "Plotting divergence" projectorder Φ nles
        lesmodel = iorder == 1 ? "DIF" : "DCF"
        fil = ifil == 1 ? "FA" : "VA"
        fig = Figure(; size = (500, 400))
        ax = Axis(
            fig[1, 1];
            yscale = islog ? log10 : identity,
            xlabel = "t",
            title = "Divergence: $lesmodel, $fil,  $nles",
        )
        lines!(ax, divergencehistory.nomodel[igrid, ifil, iorder]; label = "No closure")
        lines!(ax, divergencehistory.smag[igrid, ifil, iorder]; label = "Smagorinsky")
        lines!(ax, divergencehistory.cnn_prior[igrid, ifil, iorder]; label = "CNN (prior)")
        lines!(ax, divergencehistory.cnn_post[igrid, ifil, iorder]; label = "CNN (post)")
        lines!(
            ax,
            divergencehistory.ref[igrid, ifil, iorder];
            color = Cycled(1),
            linestyle = :dash,
            label = "Reference",
        )
        iorder == 2 && ifil == 1 && axislegend(; position = :rt)
        islog && ylims!(ax, (T(1e-6), T(1e3)))
        name = "$plotdir/divergence/"
        ispath(name) || mkpath(name)
        save("$(name)/projectorder=$(projectorder)_filter=$(Φ)_nles=$(nles).pdf", fig)
        display(fig)
    end
end

########################################################################## #src

# ## Solutions at final time

ufinal = let
    s = length(params.nles), length(params.filters), length(projectorders)
    temp = ntuple(Returns(zeros(T, ntuple(Returns(0), params.D))), params.D)
    keys = [:ref, :nomodel, :smag, :cnn_prior, :cnn_post]
    ufinal = (; map(k -> k => fill(temp, s), keys)...)
    for (iorder, projectorder) in enumerate(projectorders),
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        clean()
        @info "Computing test solutions" projectorder Φ nles
        setup = setups[igrid]
        psolver = psolver_spectral(setup)
        sample = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_test[1]))
        ustart = sample.u[1] |> device
        t = sample.t
        tlims = (t[1], t[end])
        nupdate = 5
        T = eltype(ustart[1])
        solve(closure_model, θ) =
            solve_unsteady(;
                setup = (; setup..., closure_model),
                ustart,
                tlims,
                method = RKProject(params.method, projectorder),
                psolver,
                θ,
            )[1].u .|> Array
        ufinal.ref[igrid, ifil, iorder] = sample.u[end]
        ufinal.nomodel[igrid, ifil, iorder] = solve(nothing, nothing)
        ufinal.smag[igrid, ifil, iorder] =
            solve(smagorinsky_closure(setup), θ_smag[igrid, ifil, iorder])
        ufinal.cnn_prior[igrid, ifil, iorder] =
            solve(wrappedclosure(closure, setup), device(θ_cnn_prior[igrid, ifil]))
        ufinal.cnn_post[igrid, ifil, iorder] = solve(
            wrappedclosure(closure, setup),
            device(θ_cnn_post[igrid, ifil, iorder]),
        )
    end
    ufinal
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
        setup = getsetup(; params, nles)
        fields = map(
            k -> ufinal[k][igrid, ifil, iorder] |> device,
            [:ref, :nomodel, :smag, :cnn_prior, :cnn_post],
        )
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

        setup = getsetup(; params, nles)
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
        makeplot(device(ufinal.ref[igrid, ifil, iorder]), "Reference, $fil, $nles", "ref")
        makeplot(
            device(ufinal.nomodel[igrid, ifil, iorder]),
            "No closure, $fil, $nles",
            "nomodel",
        )
        makeplot(
            device(ufinal.smag[igrid, ifil, iorder]),
            "Smagorinsky, $lesmodel, $fil, $nles",
            "smag",
        )
        makeplot(
            device(ufinal.cnn_prior[igrid, ifil, iorder]),
            "CNN (prior), $lesmodel, $fil, $nles",
            "prior",
        )
        makeplot(
            device(ufinal.cnn_post[igrid, ifil, iorder]),
            "CNN (post), $lesmodel, $fil, $nles",
            "post",
        )
    end
end
