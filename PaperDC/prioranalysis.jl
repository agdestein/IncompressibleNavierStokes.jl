using GLMakie
using CairoMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    momentum,
    momentum!,
    divergence!,
    project,
    project!,
    apply_bc_u!,
    spectral_stuff,
    kinetic_energy,
    interpolate_u_p
using NeuralClosure
using Printf
using FFTW
using PaperDC

# Output directory
output = "output/divergence"
output = "../SupervisedClosure/figures"

# Array type
ArrayType = Array
# using CUDA; ArrayType = CuArray;
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using CUDA;
ArrayType = CuArray;
CUDA.allowscalar(false);

set_theme!()
set_theme!(; GLMakie = (; scalefactor = 1.5))

# 3D
T = Float32
Re = T(2_000)
ndns = 512
D = 3
kp = 5
Δt = T(1e-4)
filterdefs = [
    (FaceAverage(), 32),
    (FaceAverage(), 64),
    (FaceAverage(), 128),
    (VolumeAverage(), 32),
    (VolumeAverage(), 64),
    (VolumeAverage(), 128),
]

# 2D 
T = Float64;
Re = T(10_000)
ndns = 1024
D = 2
kp = 20
Δt = T(5e-5)
filterdefs = [
    (FaceAverage(), 64),
    (FaceAverage(), 128),
    (FaceAverage(), 256),
    (VolumeAverage(), 64),
    (VolumeAverage(), 128),
    (VolumeAverage(), 256),
]

# Setup
lims = T(0), T(1)
dns = Setup(ntuple(α -> LinRange(lims..., ndns + 1), D)...; Re, ArrayType);
filters = map(filterdefs) do (Φ, nles)
    compression = ndns ÷ nles
    setup = Setup(ntuple(α -> LinRange(lims..., nles + 1), D)...; Re, ArrayType)
    psolver = SpectralPressureSolver(setup)
    (; setup, Φ, compression, psolver)
end;
psolver_dns = SpectralPressureSolver(dns);

# Create random initial conditions
u₀ = random_field(dns, T(0); kp, psolver = psolver_dns);

state = (; u = u₀, t = T(0));

GC.gc()
CUDA.reclaim()

energy_spectrum_plot((; u = u₀, t = T(0)); setup = dns)

fieldplot(
    (; u = u₀, t = T(0));
    setup = dns,
    # type = image,
    # colormap = :viridis,
)

# Solve unsteady problem
@time state, outputs = solve_unsteady(
    dns,
    u₀,
    # state.u,
    (T(0), T(1e-1));
    Δt,
    # Δt = T(1e-4),
    # Δt = T(1e-5),
    docopy = true,
    psolver = psolver_dns,
    processors = (
        # anim = animator(; path = "$output/solution.mkv",
        # rtp = realtimeplotter(;
        #     setup = dns,
        #     nupdate = 50,
        #     fieldname = :eig2field,
        #     levels = LinRange(T(2), T(10), 10),
        #     # levels = 5,
        #     docolorbar = false,
        # ),
        obs = observe_u(dns, psolver_dns, filters; nupdate = 20),
        # ehist = realtimeplotter(;
        #     setup,
        #     plot = energy_history_plot,
        #     nupdate = 10,
        #     displayfig = false,
        # ),
        # espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        # anim = animator(; path = "$output/solution_Re$(Int(Re)).mkv", nupdate = 10,
        #     setup = dns,
        #     fieldname = :eig2field,
        #     levels = LinRange(T(2), T(10), 10),
        #     # levels = LinRange(T(4), T(12), 10),
        #     # levels = LinRange(-1.0f0, 3.0f0, 5),
        #     # levels = LinRange(-2.0f0, 2.0f0, 5),
        #     # levels = 5,
        #     docolorbar = false,
        #     # size = (800, 800),
        #     size = (600, 600),
        # ),
        # vtk = vtk_writer(; setup, nupdate = 10, dir = output, filename = "solution"),
        # field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 5),
    ),
);
GC.gc()
CUDA.reclaim()

# 103.5320324

state.u

state.u[2]

fil = filters[2];
apply_bc_u!(state.u, T(0), dns);
v = fil.Φ(state.u, fil.setup, fil.compression);
apply_bc_u!(v, T(0), fil.setup);
Fv = momentum(v, T(0), fil.setup);
apply_bc_u!(Fv, T(0), fil.setup);
PFv = project(Fv, fil.setup; psolver = fil.psolver);
apply_bc_u!(PFv, T(0), fil.setup);
F = momentum(state.u, T(0), dns);
apply_bc_u!(F, T(0), dns);
PF = project(F, dns; psolver = psolver_dns);
apply_bc_u!(PF, T(0), dns);
ΦPF = fil.Φ(PF, fil.setup, fil.compression);
apply_bc_u!(ΦPF, T(0), fil.setup);
c = ΦPF .- PFv
apply_bc_u!(c, T(0), fil.setup)

with_theme(; fontsize = 25) do
    fig = fieldplot(
        (; u = u₀, t = T(0));
        setup = dns,
        # type = image,
        # colormap = :viridis,
        docolorbar = false,
        size = (500, 500),
        title = "u₀",
    )
    save("$output/priorfields/ustart.png", fig)
    fig = fieldplot(
        state,
        setup = dns,
        # type = image,
        # colormap = :viridis,
        docolorbar = false,
        size = (500, 500),
        title = "u",
    )
    save("$output/priorfields/u.png", fig)
    fig = fieldplot(
        (; u = v, t = T(0));
        fil.setup,
        # type = image,
        # colormap = :viridis,
        # fieldname = 1,
        docolorbar = false,
        size = (500, 500),
        # title = "ubar"
        title = "ū",
    )
    save("$output/priorfields/v.png", fig)
    fig = fieldplot(
        (; u = PF, t = T(0));
        setup = dns,
        # type = image,
        # colormap = :viridis,
        # fieldname = 1,
        docolorbar = false,
        size = (500, 500),
        title = "PF(u)",
    )
    save("$output/priorfields/PFu.png", fig)
    fig = fieldplot(
        (; u = PFv, t = T(0));
        fil.setup,
        # type = image,
        # colormap = :viridis,
        # fieldname = 1,
        docolorbar = false,
        size = (500, 500),
        # title = "PF(ubar)"
        title = "P̄F̄(ū)",
    )
    save("$output/priorfields/PFv.png", fig)
    fig = fieldplot(
        (; u = ΦPF, t = T(0));
        fil.setup,
        # type = image,
        # colormap = :viridis,
        # fieldname = 1,
        docolorbar = false,
        size = (500, 500),
        title = "ΦPF(u)",
    )
    save("$output/priorfields/PhiPFu.png", fig)
    fig = fieldplot(
        (; u = c, t = T(0));
        fil.setup,
        # type = image,
        # colormap = :viridis,
        # fieldname = 1,
        # fieldname = :velocity,
        docolorbar = false,
        size = (500, 500),
        # title = "c(u, ubar)"
        title = "c(u, ū)",
    )
    save("$output/priorfields/c.png", fig)
end

####################################################################

fieldplot(
    # (; u = u₀, t = T(0));
    state;
    setup = dns,
    fieldname = :eig2field,
    # levels = LinRange(T(2), T(10), 10),
    levels = LinRange(T(4), T(12), 10),
    # levels = LinRange(-1.0f0, 3.0f0, 5),
    # levels = LinRange(-2.0f0, 2.0f0, 5),
    # levels = 5,
    docolorbar = false,
    # size = (800, 800),
    size = (600, 600),
)

fname = "$output/prioranalysis/lambda2/Re$(Int(Re))_start.png"
fname = "$output/prioranalysis/lambda2/Re$(Int(Re))_end.png"
save(fname, current_figure())
run(`convert $fname -trim $fname`) # Requires imagemagick

i = 3
fieldplot(
    (; u = filters[i].Φ(state.u, filters[i].setup, filters[i].compression), t = T(0));
    setup = filters[i].setup,
    fieldname = :eig2field,
    # levels = LinRange(T(2), T(10), 10),
    levels = LinRange(T(4), T(12), 10),
    # levels = LinRange(-1.0f0, 3.0f0, 5),
    # levels = LinRange(-2.0f0, 2.0f0, 5),
    # levels = 5,
    docolorbar = false,
    # size = (800, 800),
    size = (600, 600),
)

fname = "$output/prioranalysis/lambda2/Re$(Int(Re))_end_filtered.png"
save(fname, current_figure())
run(`convert $fname -trim $fname`) # Requires imagemagick

field = IncompressibleNavierStokes.eig2field(state.u, dns)[dns.grid.Ip]
hist(vec(Array(log.(max.(eps(T), .-field)))))
field = nothing

i = 2
field = IncompressibleNavierStokes.eig2field(
    filters[i].Φ(state.u, filters[i].setup, filters[i].compression),
    filters[i].setup,
)[filters[i].setup.grid.Ip]
hist(vec(Array(log.(max.(eps(T), .-field)))))
field = nothing

energy_spectrum_plot(state; setup = dns)

save("spectrum.png", current_figure())

i = 6
ubar = filters[i].Φ(state.u, filters[i].setup, filters[i].compression)
energy_spectrum_plot((; u = ubar, t = T(0)); filters[i].setup)

state.u

# Float32, 1024^2:
#
# 5.711019 seconds (46.76 M allocations: 2.594 GiB, 4.59% gc time, 2.47% compilation time)
# 5.584943 seconds (46.60 M allocations: 2.583 GiB, 4.43% gc time)

# Float64, 1024^2:
#
# 9.584393 seconds (46.67 M allocations: 2.601 GiB, 2.93% gc time)
# 9.672491 seconds (46.67 M allocations: 2.601 GiB, 2.93% gc time)

# Float64, 4096^2:
#
# 114.006495 seconds (47.90 M allocations: 15.499 GiB, 0.28% gc time)
# 100.907239 seconds (46.45 M allocations: 2.588 GiB, 0.26% gc time)

# Float32, 512^3:
#
# 788.762194 seconds (162.34 M allocations: 11.175 GiB, 0.12% gc time)

9.584393 / 5.711019
9.672491 / 5.584943

# 1.0 * nbyte(Float32) * N * α * (u0 + ui + k1,k2,k3,k4 + p + maybe(complexFFT(Lap)) + maybe(boundaryfree p))
1.0 * 4 * 1024^3 * 3 * (1 + 1 + 4 + 1 / 3 + 0 * 1 * 2 + 0 * 1 / 3) # RK4: 81.6GB (111GB)
1.0 * 4 * 1024^3 * 3 * (1 + 0 + 1 + 1 / 3 + 0 * 1 * 2 + 0 * 1 / 3) # RK1: 30.0GB (60.1 GB)

1.0 * 4 * 512^3 * 3 * (1 + 1 + 4 + 1 / 3 + 0 * 1 * 2 + 0 * 1 / 3) # RK4: 10.2GB (13.9GB)
1.0 * 4 * 512^3 * 3 * (1 + 0 + 1 + 1 / 3 + 1 * 1 * 2 + 1 * 1 / 3) # RK1: 3.76GB (7.52GB)

1.0 * 8 * 512^3 * 3 * (1 + 1 + 3 + 1 / 3 + 0 * 1 * 2 + 0 * 1 / 3) # RK4: 10.2GB (13.9GB)

begin
    println("Φ\t\tM\tDu\tPv\tPc\tc")
    for o in outputs.obs
        nt = length(o.t)
        Dv = sum(o.Dv) / nt
        Pc = sum(o.Pc) / nt
        Pv = sum(o.Pv) / nt
        c = sum(o.c) / nt
        @printf(
            "%s\t%d^%d\t%.2g\t%.2g\t%.2g\t%.2g\n",
            # "%s &\t\$%d^%d\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$\n",
            typeof(o.Φ),
            o.Mα,
            D,
            Dv,
            Pv,
            Pc,
            c
        )
    end
end;

(; u, t) = state;

o = outputs.obs[1]
o.Dv
o.Pc
o.Pv
o.c

# apply_bc_u!(u, t, dns)
ubar = FaceAverage()(u, les, comp);
ubar = VolumeAverage()(u, les, comp);
# apply_bc_u!(ubar, t, les)
fieldplot((; u = ubar, t); setup = les)
fieldplot((; u, t); setup = dns)

IncompressibleNavierStokes.apply_bc_u!(ubar, t, les)
div = IncompressibleNavierStokes.divergence(ubar, les)[les.grid.Ip]

norm(div)
norm(ubar[1][les.grid.Iu[1]])

GLMakie.activate!()

# To free up memory
psolver_dns = nothing
fig = lines([1, 2, 3])
GC.gc()
CUDA.reclaim()

using CairoMakie
CairoMakie.activate!()

filters = map(filters) do f
    (; f.Φ, f.setup, f.compression)
end

# Plot predicted spectra
fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    fields = [state.u, u₀, (f.Φ(state.u, f.setup, f.compression) for f in filters)...]
    setups = [dns, dns, (f.setup for f in filters)...]
    specs = map(fields, setups) do u, setup
        GC.gc()
        CUDA.reclaim()
        (; dimension, xp, Ip) = setup.grid
        T = eltype(xp[1])
        D = dimension()
        K = size(Ip) .÷ 2
        # up = interpolate_u_p(u, setup)
        up = u
        e = sum(up) do u
            u = u[Ip]
            uhat = fft(u)[ntuple(α -> 1:K[α], D)...]
            # abs2.(uhat)
            abs2.(uhat) ./ (2 * prod(size(u))^2)
            # abs2.(uhat) ./ size(u, 1)
        end
        (; A, κ, K) = spectral_stuff(setup)
        e = A * reshape(e, :)
        # e = max.(e, eps(T)) # Avoid log(0)
        ehat = Array(e)
        (; κ, ehat)
    end
    kmax = maximum(specs[1].κ)
    # Build inertial slope above energy
    if D == 2
        # krange = [T(kmax)^(T(1) / 2), T(kmax)]
        # krange = [T(50), T(400)]
        krange = [T(16), T(128)]
    elseif D == 3
        # krange = [T(kmax)^(T(1.5) / 3), T(kmax)]
        # krange = [T(64), T(256)]
        # krange = [T(8), T(64)]
        krange = [T(16), T(100)]
    end
    slope, slopelabel =
        D == 2 ? (-T(3), L"$\kappa^{-3}") : (-T(5 / 3), L"$\kappa^{-5/3}")
    # slope, slopelabel = D == 2 ? (-T(3), "|k|⁻³") : (-T(5 / 3), "|k|⁻⁵³")
    # slope, slopelabel = D == 2 ? (-T(3), "κ⁻³") : (-T(5 / 3), "κ⁻⁵³")
    slopeconst = maximum(specs[1].ehat ./ specs[1].κ .^ slope)
    offset = D == 2 ? 3 : 2
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
        title = "Kinetic energy ($(D)D)",
    )
    # plotparts(i) = 1:specs[i].kmax, specs[i].ehat
    # plotparts(i) = 1:specs[i].kmax+1, [specs[i].ehat; eps(T)]
    plotparts(i) = specs[i].κ, specs[i].ehat
    # lines!(ax, 1:specs[1].kmax, specs[1].ehat; color = Cycled(1), label = "DNS")
    # lines!(ax, 1:specs[2].kmax, specs[2].ehat; color = Cycled(4), label = "DNS, t = 0")
    lines!(ax, plotparts(1)...; color = Cycled(1), label = "DNS")
    lines!(ax, plotparts(2)...; color = Cycled(4), label = "DNS, t = 0")
    lines!(ax, plotparts(3)...; color = Cycled(2), label = "Filtered DNS (FA)")
    lines!(ax, plotparts(4)...; color = Cycled(2))
    lines!(ax, plotparts(5)...; color = Cycled(2))
    lines!(ax, plotparts(6)...; color = Cycled(3), label = "Filtered DNS (VA)")
    lines!(ax, plotparts(7)...; color = Cycled(3))
    lines!(ax, plotparts(8)...; color = Cycled(3))
    lines!(ax, krange, inertia; color = Cycled(1), label = slopelabel, linestyle = :dash)
    D == 2 && axislegend(ax; position = :lb)
    # D == 3 && axislegend(ax; position = :rt)
    D == 3 && axislegend(ax; position = :lb)
    autolimits!(ax)
    if D == 2
        # limits!(ax, (T(0.68), T(520)), (T(1e-3), T(3e1)))
        # limits!(ax, (ax.xaxis.attributes.limits[][1], T(1000)), (T(1e-15), ax.yaxis.attributes.limits[][2]))
        limits!(ax, (T(0.8), T(800)), (T(1e-10), T(1)))
    elseif D == 3
        # limits!(ax, ax.xaxis.attributes.limits[], (T(1e-1), T(3e1)))
        # limits!(ax, ax.xaxis.attributes.limits[], (T(1e-3), ax.yaxis.attributes.limits[][2]))
        # limits!(ax, ax.xaxis.attributes.limits[], (T(3e-3), T(2e0)))
        # limits!(ax, (T(8e-1), T(400)), (T(2e-3), T(1.5e0)))
        limits!(ax, (T(8e-1), T(200)), (T(4e-5), T(1.5e0)))
    end
    fig
end
GC.gc()
CUDA.reclaim()

# save("$output/prioranalysis/spectra_$(D)D_linear_Re$(Int(Re)).pdf", fig)
save("$output/prioranalysis/spectra_$(D)D_dyadic_Re$(Int(Re)).pdf", fig)
