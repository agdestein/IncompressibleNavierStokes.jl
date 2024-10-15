# # A-priori analysis: Filtered DNS (2D or 3D)
#
# This script is used to generate results for the the paper [Agdestein2024](@citet).
#
# - Generate filtered DNS data
# - Compute quantities for different filters

# ## Load packages

if false                      #src
    include("src/PaperDC.jl") #src
end                           #src

using CairoMakie
using CUDA
using FFTW
using GLMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes: momentum, project, apply_bc_u!, spectral_stuff
using JLD2
using NeuralClosure
using PaperDC
using Printf
using Random

# Output directory
output = joinpath(@__DIR__, "output", "prioranalysis")
ispath(output) || mkpath(output)

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
    (;
        name = "2D_double",
        D = 2,
        T,
        ndns = 2048,
        Re = T(10_000),
        kp = 20,
        tlims = (T(0), T(1e-1)),
        # Δt = T(5e-5)
        filterdefs = [
            (FaceAverage(), 64),
            (FaceAverage(), 128),
            (FaceAverage(), 256),
            (VolumeAverage(), 64),
            (VolumeAverage(), 128),
            (VolumeAverage(), 256),
        ],
    )
end

# 3D configuration
case = let
    D = 3
    T, ndns = Float64, 512 # Works on a 40GB A100 GPU. Use Float32 and smaller n for less memory.
    Re = T(2_000)
    kp = 5
    tlims = (T(0), T(1e-1))
    filterdefs = [
        (FaceAverage(), 32),
        (FaceAverage(), 64),
        (FaceAverage(), 128),
        (VolumeAverage(), 32),
        (VolumeAverage(), 64),
        (VolumeAverage(), 128),
    ]
    name = "D=$(D)_T=$(T)_Re=$(Re)"
    (; D, T, ndns, Re, kp, tlims, filterdefs, name)
end

# Larger 3D configuration
case = let
    D = 3
    T = Float32
    ndns = 1024 # Works on a 80GB H100 GPU. Use smaller n for less memory.
    Re = T(6_000)
    kp = 10
    tlims = (T(0), T(1e-1))
    filterdefs = [
        (FaceAverage(), 32),
        (FaceAverage(), 64),
        (FaceAverage(), 128),
        (FaceAverage(), 256),
        (VolumeAverage(), 32),
        (VolumeAverage(), 64),
        (VolumeAverage(), 128),
        (VolumeAverage(), 256),
    ]
    name = "D=$(D)_T=$(T)_Re=$(Re)"
    (; D, T, ndns, Re, kp, tlims, filterdefs, name)
end

# Setup
lims = case.T(0), case.T(1)
dns = let
    setup = Setup(;
        x = ntuple(α -> LinRange(lims..., case.ndns + 1), case.D),
        case.Re,
        ArrayType,
    )
    psolver = default_psolver(setup)
    (; setup, psolver)
end;
filters = map(case.filterdefs) do (Φ, nles)
    compression = case.ndns ÷ nles
    setup = Setup(;
        x = ntuple(α -> LinRange(lims..., nles + 1), case.D),
        case.Re,
        ArrayType,
    )
    psolver = default_psolver(setup)
    (; setup, Φ, compression, psolver)
end;

# Create random initial conditions
rng = Xoshiro(12345)
ustart = random_field(dns.setup, case.T(0); case.kp, dns.psolver, rng);
clean()

# Solve unsteady problem
state, outputs = solve_unsteady(;
    dns.setup,
    ustart,
    case.tlims,
    # Δt,
    docopy = true, # leave initial conditions unchanged, false to free up memory
    dns.psolver,
    processors = (
        obs = observe_u(dns.setup, dns.psolver, filters; nupdate = 20),
        log = timelogger(; nupdate = 5),
    ),
);
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
GLMakie.activate!()

# Make plots
D == 3 && with_theme() do
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

open("$output/$(case.name)_averages.txt", "w") do io
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
    fields = [state.u, ustart, map(f -> f.Φ(state.u, f.setup, f.compression), filters)...]
    setups = [dns.setup, dns.setup, getfield.(filters, :setup)...]
    specs = map(fields, setups) do u, setup
        clean() # Free up memory
        state = (; u)
        spec = observespectrum(state; setup)
        (; spec.κ, ehat = spec.ehat[])
    end
    save_object("$output/$(case.name)_spectra.jld2", specs)
end

specs = load_object("$output/$(case.name)_spectra.jld2")

# Plot predicted spectra
CairoMakie.activate!()
with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ff9900"])) do
    (; D, T) = case
    kmax = maximum(specs[1].κ)
    ## Build inertial slope above energy
    krange, slope, slopelabel = if D == 2
        [T(16), T(128)], -T(3), L"$\kappa^{-3}$"
    elseif D == 3
        [T(16), T(100)], -T(5 / 3), L"$\kappa^{-5/3}$"
    end
    slopeconst = maximum(specs[1].ehat ./ specs[1].κ .^ slope)
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
    lines!(ax, plotparts(1)...; color = Cycled(1), label = "DNS")
    lines!(ax, plotparts(2)...; color = Cycled(4), label = "DNS, t = 0")
    lines!(ax, plotparts(3)...; color = Cycled(2), label = "Filtered DNS (FA)")
    lines!(ax, plotparts(4)...; color = Cycled(2))
    lines!(ax, plotparts(5)...; color = Cycled(2))
    lines!(ax, plotparts(6)...; color = Cycled(3), label = "Filtered DNS (VA)")
    lines!(ax, plotparts(7)...; color = Cycled(3))
    lines!(ax, plotparts(8)...; color = Cycled(3))
    lines!(ax, krange, inertia; color = Cycled(1), label = slopelabel, linestyle = :dash)
    axislegend(ax; position = :lb)
    autolimits!(ax)
    if D == 2
        limits!(ax, (T(0.8), T(800)), (T(1e-10), T(1e0)))
        # limits!(ax, (T(16), T(128)), (T(1e-4), T(1e-1)))
    elseif D == 3
        limits!(ax, (T(8e-1), T(200)), (T(4e-5), T(1.5e0)))
    end

    x1, y1 = 477, 358
    x0, y0 = x1 - 90, y1 - 94

    k0, k1 = 100, 130
    e0, e1 = 6e-5, 3e-4
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
        yscale = log10,
        yticksvisible = false,
        yticklabelsvisible = false,
        xticksvisible = false,
        xticklabelsvisible = false,
        xgridvisible = false,
        ygridvisible = false,
        backgroundcolor = :white,
    )
    # https://discourse.julialang.org/t/makie-inset-axes-and-their-drawing-order/60987/5
    translate!(ax2.scene, 0, 0, 10)
    translate!(ax2.elements[:background], 0, 0, 9)
    lines!(ax2, plotparts(1)...; color = Cycled(1))
    lines!(ax2, plotparts(5)...; color = Cycled(2))
    lines!(ax2, plotparts(8)...; color = Cycled(3))

    save("$output/$(case.name)_spectra.pdf", fig)
    fig
end
clean()
