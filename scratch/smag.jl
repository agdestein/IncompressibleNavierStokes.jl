# # DNS, filtered DNS, and LES
#
# In this example we compare DNS, filtered DNS, and LES.

if false                                            #src
    include("../src/IncompressibleNavierStokes.jl") #src
end                                                 #src

using CairoMakie
# using GLMakie
using IncompressibleNavierStokes
using NeuralClosure
using FFTW
using CUDA

const INS = IncompressibleNavierStokes

# Setups
Φ = FaceAverage()
compression = 2
n_dns = 256
n_les = div(n_dns, compression)
dns, les = map((n_dns, n_les)) do n
    ax = LinRange(0.0, 1.0, n + 1)
    setup = Setup(; x = (ax, ax, ax), Re = 3e3, ArrayType = CuArray)
    psolver = default_psolver(setup)
    ustart = random_field(setup, 0.0)
    (; setup, psolver, ustart)
end;

Φ(les.ustart, dns.ustart, les.setup, compression);

# fieldplot((; u = dns.ustart, temp = nothing, t = 0.0); dns.setup)
# fieldplot((; u = les.ustart, temp = nothing, t = 0.0); les.setup)

θ = 0.17
les_smag =
    (; les..., θ, setup = (; les.setup..., closure_model = smagorinsky_closure(les.setup)))
les_smag_nat = (;
    les...,
    θ,
    setup = (; les.setup..., closure_model = INS.smagorinsky_closure_natural(les.setup)),
)

tlims = (0.0, 1.0)
processors = (; log = timelogger(; nupdate = 10))
state_dns, _ = solve_unsteady(; dns..., tlims, processors);
state_nomodel, _ = solve_unsteady(; les..., tlims, processors);
state_smag, _ = solve_unsteady(; les1..., tlims, processors);
state_smag_nat, _ = solve_unsteady(; les2..., tlims, processors);

energy_spectrum_plot(state_nomodel; les.setup)

spectra = map([
    (state_dns, dns.setup, "DNS"),
    (state_nomodel, les.setup, "No closure"),
    (state_smag, les.setup, "Smagorinsky"),
    (state_smag_nat, les.setup, "Smagorinsky (natural tensor positions)"),
]) do (state, setup, label)
    (; ehat, κ) = observespectrum(state; setup)
    (; κ, ehat, label)
end

# Plot predicted spectra
let
    D = dns.setup.grid.dimension()
    kmax = maximum(spectra[1].κ)
    ## Build inertial slope above spectrum
    krange, slope, slopelabel = if D == 2
        [16.0, 128.0], -3.0, L"$\kappa^{-3}$"
    elseif D == 3
        [8.0, 32.0], -5.0 / 3, L"$\kappa^{-5/3}$"
    end
    slopeconst = maximum(spectra[1].ehat[] ./ spectra[1].κ .^ slope)
    offset = D == 2 ? 3 : 2
    inertia = offset .* slopeconst .* krange .^ slope
    ## Nice ticks
    logmax = round(Int, log2(kmax + 1))
    xticks = 2.0 .^ (0:logmax)
    ## Make plot
    fig = Figure(;
    # size = (500, 400),
    # size = (1200, 900),
    )
    ax = Axis(
        fig[1, 1];
        xticks,
        xlabel = "κ",
        xscale = log10,
        yscale = log10,
        limits = (1, kmax, 1e-8, 1.0),
        title = "Kinetic energy ($(D)D)",
    )
    for (i, s) in enumerate(spectra)
        lines!(ax, s.κ, s.ehat; color = Cycled(i), s.label)
    end
    lines!(ax, krange, inertia; color = Cycled(1), label = slopelabel, linestyle = :dash)
    axislegend(ax; position = :lb)
    # Legend(fig[2, 1], ax; orientation = :horizontal, framevisible = false)
    autolimits!(ax)
    if D == 2
        fimits!(ax, (0.8, 800.0), (1e-10, 1.0))
    elseif D == 3
        limits!(ax, (8e-1, 100.0), (1e-5, 1e0))
    end
    # path = "$output/prioranalysis"
    # ispath(path) || mkpath(path)
    # save("$path/spectra_$(D)D_dyadic_Re$(Int(Re)).pdf", fig)
    fig
end
clean()
