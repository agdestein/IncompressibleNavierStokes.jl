# # A-priori analysis: Filtered DNS (2D or 3D)
#
# This script is used to generate results for the the paper [Agdestein2024](@citet).
#
# - Generate filtered DNS data
# - Compute quantities for different filters

@info "Script started"

if false                      #src
    include("src/PaperDC.jl") #src
end                           #src

# Output directory
output = joinpath(@__DIR__, "output", "prioranalysis")
logdir = joinpath(output, "logs")
ispath(output) || mkpath(output)
ispath(logdir) || mkpath(logdir)

# ## Configure logger

using PaperDC
using Dates

# Write output to file, as the default SLURM file is not updated often enough
isslurm = haskey(ENV, "SLURM_JOB_ID")
if isslurm
    jobid = parse(Int, ENV["SLURM_JOB_ID"])
    logfile = "job=$(jobid)_$(Dates.now()).out"
else
    logfile = "log_$(Dates.now()).out"
end
logfile = joinpath(logdir, logfile)
setsnelliuslogger(logfile)

@info "# A-posteriori analysis: Forced turbulence (2D)"

# ## Load packages

using CairoMakie
using CUDA
using FFTW
# using GLMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes: apply_bc_u!, ode_method_cache
using JLD2
using NeuralClosure
using PaperDC
using Printf
using Random

# ## Hardware selection

if CUDA.functional()
    ## For running on GPU
    CUDA.allowscalar(false)
    ArrayType = CuArray
    clean() = (GC.gc(); CUDA.reclaim()) # This seems to be needed to free up memory
else
    ## For running on CPU
    ArrayType = Array
    clean() = nothing
end

# ## Setup

# 2D configuration
case = let
    T = Float64
    D = 2
    ndns = 4096
    Re = T(1e4)
    kp = 20
    tlims = (T(0), T(1e-1))
    docopy = true
    nles = [32, 64, 128, 256]
    filterdefs = [FaceAverage(), VolumeAverage()]
    name = "D=$(D)_T=$(T)_Re=$(Re)_t=$(tlims[2])"
    (; D, T, ndns, Re, kp, tlims, docopy, nles, filterdefs, name)
end

# 3D configuration
case = let
    D = 3
    T = Float32
    ndns = 1024 # Works on a 80GB H100 GPU. Use smaller n for less memory.
    Re = T(1e4)
    kp = 20
    tlims = (T(0), T(1e-1))
    docopy = false
    nles = [32, 64, 128, 256]
    filterdefs = [FaceAverage(), VolumeAverage()]
    name = "D=$(D)_T=$(T)_Re=$(Re)_t=$(tlims[2])"
    (; D, T, ndns, Re, kp, tlims, docopy, nles, filterdefs, name)
end

# Setup
lims = case.T(0), case.T(1)
dns = let
    setup = Setup(; x = ntuple(α -> range(lims..., case.ndns + 1), case.D), case.Re, ArrayType)
    psolver = default_psolver(setup)
    (; setup, psolver)
end;
filters = map(Iterators.product(case.nles, case.filterdefs)) do (nles, Φ)
    compression = case.ndns ÷ nles
    setup =
        Setup(; x = ntuple(α -> range(lims..., nles + 1), case.D), case.Re, ArrayType)
    psolver = default_psolver(setup)
    (; setup, Φ, compression, psolver)
end;

# Create random initial conditions
rng = Xoshiro(12345)
ustart = random_field(dns.setup, case.T(0); case.kp, dns.psolver, rng);
clean()

# Compute initial spectrum since we will overwrite ustart to save memory
specstart = let
    state = (; u = ustart)
    spec = observespectrum(state; dns.setup)
    (; spec.κ, ehat = spec.ehat[])
end
clean()

# Solve unsteady problem
state, outputs = let
    method = RKMethods.Wray3(; case.T)
    cache = ode_method_cache(method, dns.setup)
    solve_unsteady(;
        dns.setup,
        ustart,
        case.tlims,
        case.docopy, # leave initial conditions unchanged, false to free up memory
        method,
        cache,
        dns.psolver,
        processors = (
            obs = observe_u(
                dns.setup,
                dns.psolver,
                filters;
                PF = cache.ku[1],
                p = cache.p,
                nupdate = 20,
            ),
            log = timelogger(; nupdate = 5),
        ),
    )
end;
clean()

# ## Plot 2D fields

case.D == 2 && with_theme(; fontsize = 25) do
    (; T) = case
    ## Compute quantities
    fil = filters[2]
    apply_bc_u!(state.u, T(0), dns.setup)
    Φu = fil.Φ(state.u, fil.setup, fil.compression)
    apply_bc_u!(Φu, T(0), fil.setup)
    Fv = momentum(Φu, nothing, T(0), fil.setup)
    apply_bc_u!(Fv, T(0), fil.setup)
    PFv = project(Fv, fil.setup; psolver = fil.psolver)
    apply_bc_u!(PFv, T(0), fil.setup)
    F = momentum(state.u, nothing, T(0), dns.setup)
    apply_bc_u!(F, T(0), dns.setup)
    PF = project(F, dns.setup; dns.psolver)
    apply_bc_u!(PF, T(0), dns.setup)
    ΦPF = fil.Φ(PF, fil.setup, fil.compression)
    apply_bc_u!(ΦPF, T(0), fil.setup)
    c = ΦPF .- PFv
    apply_bc_u!(c, T(0), fil.setup)

    ## Make plots
    makeplot(field, setup, title, name) = save(
        "$output/$(case.name)_$name.png",
        fieldplot(
            (; u = field, temp = nothing, t = T(0));
            setup,
            title,
            docolorbar = false,
            size = (500, 500),
        ),
    )
    makeplot(ustart, dns.setup, "u₀", "ustart")
    makeplot(state.u, dns.setup, "u", "u")
    makeplot(Φu, fil.setup, "ū", "Phi_u")
    makeplot(PF, dns.setup, "PF(u)", "P_F_u")
    makeplot(PFv, fil.setup, "P̄F̄(ū)", "P_F_Phi_u")
    makeplot(ΦPF, fil.setup, "ΦPF(u)", "Phi_P_F_u")
    makeplot(c, fil.setup, "c(u)", "c")
end

# ## Plot 3D fields

# Contour plots in 3D only work with GLMakie.
# For using GLMakie on headless servers, see
# <https://docs.makie.org/stable/explanations/headless/#glmakie>
# GLMakie.activate!()

# Make plots
# D == 3 &&
false && with_theme() do
    function makeplot(field, setup, name)
        name = "$output/$(case.name)_$name.png"
        save(
            name,
            fieldplot(
                (; u = field, t = T(0));
                setup,
                fieldname = :eig2field,
                levels = LinRange(T(4), T(12), 10),
                docolorbar = false,
                size = (600, 600),
            ),
        )
        try
            ## Trim whitespace with ImageMagick
            run(`magick $name -trim $name`)
        catch e
            @warn """
            ImageMagick not found.
            Skipping image trimming.
            Install from <https://imagemagick.org/>.
            """
        end
    end
    makeplot(u₀, dns.setup, "Re=$(Int(Re))_start") # Requires docopy = true in solve
    makeplot(state.u, dns.setup, "Re=$(Int(Re))_end")
    i = 3
    makeplot(
        filters[i].Φ(state.u, filters[i].setup, filters[i].compression),
        filters[i].setup,
        "Re=$(Int(Re))_end_filtered",
    )
end

# ## Compute average quantities

open("$output/averages_$(case.name).txt", "w") do io
    println(io, "Φ\t\tM\tDu\tPv\tPc\tc\tE")
    for o in outputs.obs
        nt = length(o.t)
        Dv = sum(o.Dv) / nt
        Pc = sum(o.Pc) / nt
        Pv = sum(o.Pv) / nt
        c = sum(o.c) / nt
        E = sum(o.E) / nt
        @printf(
            io,
            "%s\t%d^%d\t%.2g\t%.2g\t%.2g\t%.2g\t%.2g\n",
            ## "%s &\t\$%d^%d\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$\n",
            typeof(o.Φ),
            o.Mα,
            case.D,
            Dv,
            Pv,
            Pc,
            c,
            E,
        )
    end
end

# ## Plot spectra

let
    fields = [state.u, map(f -> f.Φ(state.u, f.setup, f.compression), filters)...]
    setups = [dns.setup, getfield.(filters, :setup)...]
    specs = map(fields, setups) do u, setup
        clean() # Free up memory
        state = (; u)
        spec = observespectrum(state; setup)
        (; spec.κ, ehat = spec.ehat[])
    end
    pushfirst!(specs, specstart)
    save_object("$output/spectra_$(case.name).jld2", specs)
end

specs = load_object("$output/spectra_$(case.name).jld2")
# specs = load_object("$(ENV["HOME"])/haha/$(case.name)_spectra.jld2")

# Plot predicted spectra
CairoMakie.activate!()

with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])) do
    (; D, T) = case
    kmax = maximum(specs[1].κ)
    ## Build inertial slope above energy
    krange, slope, slopelabel = if D == 2
        # [T(16), T(128)], -T(3), L"$\kappa^{-3}$"
        [T(18), T(128)], -T(3), L"$\kappa^{-3}$"
    elseif D == 3
        # [T(16), T(100)], -T(5 / 3), L"$\kappa^{-5/3}$"
        # [T(32), T(128)], -T(5 / 3), L"$\kappa^{-5/3}$"
        [T(80), T(256)], -T(5 / 3), L"$\kappa^{-5/3}$"
    end
    slopeconst = maximum(specs[2].ehat ./ specs[2].κ .^ slope)
    offset = D == 2 ? 3 : 2
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
        title = "Kinetic energy ($(D)D)",
    )
    plotparts(i) = specs[i].κ, specs[i].ehat
    nnles = length(case.nles)
    FA = 3:2+nnles
    VA = 3+nnles:2+2*nnles

    # lines!(ax, plotparts(1)...; color = Cycled(4), label = "DNS, t = 0")
    lines!(ax, plotparts(2)...; color = Cycled(1), label = "DNS")
    for i in FA
        label = i == FA[1] ? "Filtered DNS (FA)" : nothing
        lines!(ax, plotparts(i)...; color = Cycled(2), label)
    end
    for i in VA
        label = i == VA[1] ? "Filtered DNS (VA)" : nothing
        lines!(ax, plotparts(i)...; color = Cycled(3), label)
    end
    lines!(ax, krange, inertia; color = Cycled(1), label = slopelabel, linestyle = :dash)
    axislegend(
        ax;
        position = :lb,
        # position = (0.2, 0.01),
    )
    autolimits!(ax)
    if D == 2
        limits!(ax, (T(0.8), T(460)), (T(1e-7), T(1e0)))
        # limits!(ax, (T(16), T(128)), (T(1e-4), T(1e-1)))
        o = 6
        sk, se = 1.08, 1.4
        text!(ax, "4096"; position = (198, 1.4e-7))
        x1, y1 = 477, 358
        x0, y0 = x1 - 90, y1 - 94
        textk, texte = 1.5, 2.0
    elseif D == 3
        # limits!(ax, (T(8e-1), T(700)), (T(5e-9), T(3.0e-2)))
        # limits!(ax, (T(8e-1), T(850)), (T(1.5e-5), T(1.0e-1)))
        # text!(ax, "1024"; position = (241, 2.4e-8))
        text!(ax, "1024"; position = (259, 2.4e-5))
        o = 7
        sk, se = 1.15, 1.3
        x1, y1 = 390, 185
        x0, y0 = x1 - 120, y1 - 120
        textk, texte = 1.5, 1.5
    end
    kk, ee = plotparts(FA[end])
    kk, ee = kk[end-o], ee[end-o]
    k0, k1 = kk / sk, kk * sk
    e0, e1 = ee / se, ee * se
    for (i, nles) in zip(VA, case.nles)
        κ, e = plotparts(i)
        text!(ax, "$nles"; position = (κ[end] / textk, e[end] / texte))
    end

    limits = (k0, k1, e0, e1)
    lines!(
        ax,
        [
            Point2f(k0, e0),
            Point2f(k1, e0),
            Point2f(k1, e1),
            Point2f(k0, e1),
            Point2f(k0, e0),
        ];
        color = :black,
        linewidth = 1.5,
    )
    ax2 = Axis(
        fig;
        bbox = BBox(x0, x1, y0, y1),
        limits,
        xscale = log10,
        yscale = log10,
        xticksvisible = false,
        xticklabelsvisible = false,
        xgridvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        ygridvisible = false,
        backgroundcolor = :white,
    )
    # https://discourse.julialang.org/t/makie-inset-axes-and-their-drawing-order/60987/5
    translate!(ax2.scene, 0, 0, 10)
    translate!(ax2.elements[:background], 0, 0, 9)
    lines!(ax2, plotparts(2)...; color = Cycled(1))
    lines!(ax2, plotparts(FA[end])...; color = Cycled(2))
    lines!(ax2, plotparts(VA[end])...; color = Cycled(3))
    save("$output/spectra_$(case.name).pdf", fig)
    fig
end
