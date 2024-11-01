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
    case = (;
        T,
        D = 2,
        ndns = 4096,
        Re = T(1e4),
        kp = 20,
        tlims = (T(0), T(1e0)),
        docopy = true,
        bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y),
        issteadybodyforce = true,
        nles = [32, 64, 128, 256],
        filterdefs = [FaceAverage(), VolumeAverage()],
        name,
    )
    (; case..., name = "D=$(case.D)_T=$(T)_Re=$(case.Re)_t=$(case.tlims[2])")
end

# 3D configuration
case = let
    T = Float32
    case = (;
        T,
        D = 3,
        ndns = 1024, # Works on a 80GB H100 GPU. Use smaller n for less memory.
        Re = T(6e3),
        kp = 20,
        tlims = (T(0), T(1e0)),
        bodyforce = (dim, x, y, z, t) -> (dim == 1) * 5 * sinpi(8 * y),
        issteadybodyforce = true,
        docopy = false,
        nles = [32, 64, 128, 256],
        filterdefs = [FaceAverage(), VolumeAverage()],
    )
    (; case..., name = "D=$(case.D)_T=$(case.T)_Re=$(case.Re)_t=$(case.tlims[2])")
end

casedir = joinpath(output, case.name)
ispath(casedir) || mkpath(casedir)

# Setup
lims = case.T(0), case.T(1)
dns = let
    setup = Setup(;
        x = ntuple(α -> range(lims..., case.ndns + 1), case.D),
        case.Re,
        case.bodyforce,
        case.issteadybodyforce,
        ArrayType,
    )
    psolver = default_psolver(setup)
    (; setup, psolver)
end;
filters = map(Iterators.product(case.nles, case.filterdefs)) do (nles, Φ)
    compression = case.ndns ÷ nles
    setup = Setup(;
        x = ntuple(α -> range(lims..., nles + 1), case.D),
        case.Re,
        case.bodyforce,
        case.issteadybodyforce,
        ArrayType,
    )
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

# Save initial conditions
@info "Saving initial conditions"
save_object("$casedir/ustart.jld2", Array.(ustart))

# Solve unsteady problem
@info "Starting time stepping"
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

# Save final velocity
@info "Starting final velocity"
save_object("$casedir/uend.jld2", Array.(state.u))

# ## Plot 2D fields

case.D == 2 && with_theme() do
    @info "Plotting 2D fields"
    (; T) = case

    ## Compute quantities
    for fil in filters
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
        fields = [
            (ustart, dns.setup, "u₀"),
            (c, fil.setup, "c(u)"),
            (state.u, dns.setup, "u"),
            (PF, dns.setup, "PF(u)"),
            (Φu, fil.setup, "ū"),
            (PFv, fil.setup, "P̄F̄(ū)"),
        ]
        fig = Figure(; size = (600, 450))
        for (I, field) in enumerate(fields)
            f, setup, title = field
            (; Ip, xp) = setup.grid
            i, j = CartesianIndices((2, 3))[I].I
            w = vorticity(f, setup)
            # w = f[1] |> Array
            w = w[Ip] |> Array
            lims = get_lims(w)
            xw = xp[1][Ip.indices[1]], xp[2][Ip.indices[2]]
            xw = Array.(xw)
            heatmap(
                fig[i, j],
                xw...,
                w;
                colorrange = lims,
                axis = (;
                    title,
                    xticksvisible = false,
                    xticklabelsvisible = false,
                    yticksvisible = false,
                    yticklabelsvisible = false,
                    aspect = DataAspect(),
                ),
            )
        end
        display(fig)
        name = "$casedir/fields_filter=$(fil.Φ)_nles=$(fil.setup.grid.Np[1]).png"
        save(name, fig; px_per_unit = 2)
    end
end

# ## Plot 3D fields

# Contour plots in 3D only work with GLMakie.
# For using GLMakie on headless servers, see
# <https://docs.makie.org/stable/explanations/headless/#glmakie>
# GLMakie.activate!()

# Make plots
dovolumeplot = false && D == 3
dovolumeplot && with_theme() do
    @info "Plotting 3D fields"
    function makeplot(field, setup, name)
        name = "$casedir/$name.png"
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
    makeplot(u₀, dns.setup, "start") # Requires docopy = true in solve
    makeplot(state.u, dns.setup, "end")
    i = 3
    makeplot(
        filters[i].Φ(state.u, filters[i].setup, filters[i].compression),
        filters[i].setup,
        "end_filtered",
    )
end

# ## Compute average quantities

open("$casedir/averages.txt", "w") do io
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
    save_object("$casedir/spectra.jld2", specs)
end

specs = load_object("$casedir/spectra.jld2")

with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])) do
    (; D, T) = case

    ## Build inertial slope above energy
    krange, slope, slopelabel = if D == 2
        [T(8), T(50)], -T(3), L"$\kappa^{-3}$"
    elseif D == 3
        # [T(80), T(256)], -T(5 / 3), L"$\kappa^{-5/3}$"
        [T(5), T(32)], -T(5 / 3), L"$\kappa^{-5/3}$"
    end
    slopeconst = maximum(specs[2].ehat ./ specs[2].κ .^ slope)
    offset = D == 2 ? 3 : 2
    inertia = offset .* slopeconst .* krange .^ slope

    ## Nice ticks
    kmax = maximum(specs[1].κ)
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
        title = "Energy spectrum ($(D)D)",
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
        xlims!(ax, T(0.8), T(460))
        ylims!(ax, T(1e-10), T(1e0))
    elseif D == 3
        xlims!(ax, 0.8, 290)
        xlims!(ax, 0.8, 200)
        ylims!(ax, 1e-12, 3e-3)
    end

    # Add resolution numbers just below plots
    if D == 2
        text!(ax, "4096"; position = (175, 1.5e-10))
        textk, texte = 1.5, 2.0
    elseif D == 3
        # text!(ax, "1024"; position = (241, 2.4e-8))
        # text!(ax, "1024"; position = (259, 2.4e-5))
        # text!(ax, "1024"; position = (110, 1.5e-11))
        text!(ax, "1024"; position = (90, 1.3e-12))
        textk, texte = 1.55, 1.6
    end
    for (i, nles) in zip(VA, case.nles)
        κ, e = plotparts(i)
        text!(ax, "$nles"; position = (κ[end] / textk, e[end] / texte))
    end

    # Plot zoom-in box
    if D == 2
        o = 6
        sk, se = 1.06, 1.4
        x1, y1 = 477, 358
        x0, y0 = x1 - 90, y1 - 94
    elseif D == 3
        o = 7
        sk, se = 1.05, 1.3
        # x1, y1 = 360, 185
        x1, y1 = 477, 358
        x0, y0 = x1 - 90, y1 - 94
    end
    kk, ee = plotparts(FA[end])
    kk, ee = kk[end-o], ee[end-o]
    k0, k1 = kk / sk, kk * sk
    e0, e1 = ee / se, ee * se
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

    save("$casedir/spectra.pdf", fig)
    fig
end
