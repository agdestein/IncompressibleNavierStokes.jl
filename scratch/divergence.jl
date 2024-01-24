# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

#md using CairoMakie
using GLMakie
# using CairoMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    momentum!, divergence!, project!, apply_bc_u!, spectral_stuff, kinetic_energy
using LinearAlgebra
using Printf
using FFTW

# Output directory
output = "output/divergence"
output = "../SupervisedClosure/figures"

# Floating point precision
T = Float64

# Array type
ArrayType = Array
# using CUDA; ArrayType = CuArray;
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using CUDA;
T = Float64;
# T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);

set_theme!()
set_theme!(; GLMakie = (; scalefactor = 1.5))

function observe_v(dnsobs, Φ, les, compression, psolver)
    (; grid) = les
    (; dimension, N, Iu, Ip) = grid
    D = dimension()
    Mα = N[1] - 2
    v = zero.(Φ(dnsobs[].u, les, compression))
    Pv = zero.(v)
    p = zero(v[1])
    div = zero(p)
    ΦPF = zero.(v)
    PFΦ = zero.(v)
    c = zero.(v)
    T = eltype(v[1])
    results = (;
        Φ,
        Mα,
        t = zeros(T, 0),
        Dv = zeros(T, 0),
        Pv = zeros(T, 0),
        Pc = zeros(T, 0),
        c = zeros(T, 0),
    )
    on(dnsobs) do (; u, PF, t)
        push!(results.t, t)

        Φ(v, u, les, compression)
        apply_bc_u!(v, t, les)
        Φ(ΦPF, PF, les, compression)
        momentum!(PFΦ, v, t, les)
        apply_bc_u!(PFΦ, t, les; dudt = true)
        project!(PFΦ, les; psolver, div, p)
        foreach(α -> c[α] .= ΦPF[α] .- PFΦ[α], 1:D)
        apply_bc_u!(c, t, les)
        divergence!(div, v, les)
        norm_Du = norm(div[Ip])
        norm_v = sqrt(sum(α -> sum(abs2, v[α][Iu[α]]), 1:D))
        push!(results.Dv, norm_Du / norm_v)

        copyto!.(Pv, v)
        project!(Pv, les; psolver, div, p)
        foreach(α -> Pv[α] .= Pv[α] .- v[α], 1:D)
        norm_vmPv = sqrt(sum(α -> sum(abs2, Pv[α][Iu[α]]), 1:D))
        push!(results.Pv, norm_vmPv / norm_v)

        Pc = Pv
        copyto!.(Pc, c)
        project!(Pc, les; psolver, div, p)
        foreach(α -> Pc[α] .= Pc[α] .- c[α], 1:D)
        norm_cmPc = sqrt(sum(α -> sum(abs2, Pc[α][Iu[α]]), 1:D))
        norm_c = sqrt(sum(α -> sum(abs2, c[α][Iu[α]]), 1:D))
        push!(results.Pc, norm_cmPc / norm_c)

        norm_ΦPF = sqrt(sum(α -> sum(abs2, ΦPF[α][Iu[α]]), 1:D))
        push!(results.c, norm_c / norm_ΦPF)
    end
    results
end

observe_u(dns, psolver_dns, filters; nupdate = 1) =
    processor() do state
        PF = zero.(state[].u)
        div = zero(state[].u[1])
        p = zero(state[].u[1])
        dnsobs = Observable((; state[].u, PF, state[].t))
        results = [
            observe_v(dnsobs, Φ, setup, compression, psolver) for
            (; setup, Φ, compression, psolver) in filters
        ]
        on(state) do (; u, t, n)
            n % nupdate == 0 || return
            apply_bc_u!(u, t, dns)
            momentum!(PF, u, t, dns)
            apply_bc_u!(PF, t, dns; dudt = true)
            project!(PF, dns; psolver = psolver_dns, div, p)
            dnsobs[] = (; u, PF, t)
        end
        # state[] = state[] # Save initial conditions
        results
    end

# Viscosity model
Re = T(10_000)
# Re = T(6_000)

# A 2D grid is a Cartesian product of two vectors
ndns = 4096
# ndns = 256
# ndns = 512
lims = T(0), T(1)
D = 2
# D = 3
dns = Setup(ntuple(α -> LinRange(lims..., ndns + 1), D)...; Re, ArrayType);

filters = map([
    # (FaceAverage(), 32),
    (FaceAverage(), 64),
    (FaceAverage(), 128),
    (FaceAverage(), 256),
    # (VolumeAverage(), 32),
    (VolumeAverage(), 64),
    (VolumeAverage(), 128),
    (VolumeAverage(), 256),
]) do (Φ, nles)
    compression = ndns ÷ nles
    setup = Setup(ntuple(α -> LinRange(lims..., nles + 1), D)...; Re, ArrayType)
    psolver = SpectralPressureSolver(setup)
    (; setup, Φ, compression, psolver)
end;

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
psolver_dns = SpectralPressureSolver(dns);

# Create random initial conditions
# u₀ = random_field(dns, T(0); kp = 5, psolver = psolver_dns);
u₀ = random_field(dns, T(0); kp = 20, psolver = psolver_dns);
state = (; u = u₀, t = T(0));

GC.gc()
CUDA.reclaim()

energy_spectrum_plot((; u = u₀, t = T(0)); setup = dns, doaverage = false)

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
    (T(0), T(1e-1));
    # Δt = T(1e-4),
    Δt = T(5e-5),
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
        # anim = animator(; setup, path = "$output/solution.mkv", nupdate = 20),
        # vtk = vtk_writer(; setup, nupdate = 10, dir = output, filename = "solution"),
        # field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 5),
    ),
);

# 103.5320324

state.u[1]

fieldplot(
    (; u = u₀, t = T(0));
    setup = dns,
    # type = image,
    # colormap = :viridis,
    docolorbar = false,
    size = (500, 500),
)

save("$output/vorticity_start.png", current_figure())

fieldplot(
    state,
    setup = dns,
    # type = image,
    # colormap = :viridis,
    docolorbar = false,
    size = (500, 500),
)

save("$output/vorticity_end.png", current_figure())

i = 1
fieldplot(
    (; u = filters[i].Φ(state.u, filters[i].setup, filters[i].compression), t = T(0));
    setup = filters[i].setup,
    # type = image,
    # colormap = :viridis,
    docolorbar = false,
    size = (500, 500),
)

save("$output/vorticity_end_$(filters[i].compression).png", current_figure())

fieldplot(
    # (; u = u₀, t = T(0));
    state;
    setup = dns,
    fieldname = :eig2field,
    levels = LinRange(T(2), T(10), 10),
    # levels = LinRange(T(4), T(12), 10),
    # levels = LinRange(-1.0f0, 3.0f0, 5),
    # levels = LinRange(-2.0f0, 2.0f0, 5),
    # levels = 5,
    docolorbar = false,
    # size = (800, 800),
    size = (500, 500),
)

save("$output/lambda2_start.png", current_figure())
save("$output/lambda2_end.png", current_figure())

i = 2
fieldplot(
    (; u = filters[i].Φ(state.u, filters[i].setup, filters[i].compression), t = T(0));
    setup = filters[i].setup,
    fieldname = :eig2field,
    levels = LinRange(T(2), T(10), 10),
    # levels = LinRange(T(4), T(12), 10),
    # levels = LinRange(-1.0f0, 3.0f0, 5),
    # levels = LinRange(-2.0f0, 2.0f0, 5),
    # levels = 5,
    docolorbar = false,
    # size = (800, 800),
    size = (500, 500),
)

save("$output/lambda2_end_filtered.png", current_figure())

i = 2
# field = IncompressibleNavierStokes.eig2field(state.u, dns)[dns.grid.Ip]
field = IncompressibleNavierStokes.eig2field(
    filters[i].Φ(state.u, filters[i].setup, filters[i].compression),
    filters[i].setup,
)[filters[i].setup.grid.Ip]
# hist(vec(Array(log(max(eps(T), field)))
hist(vec(Array(log.(max.(eps(T), .-field)))))
field = nothing

energy_spectrum_plot(state; setup = dns, doaverage = false)

i = 6
ubar = filters[i].Φ(state.u, filters[i].setup, filters[i].compression)
energy_spectrum_plot((; u = ubar, t = T(0)); filters[i].setup, doaverage = false)

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
            # "%s\t%d^%d\t%.2g\t%.2g\t%.2g\t%.2g\n",
            "%s &\t\$%d^%d\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$\n",
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

using CairoMakie
CairoMakie.activate!()

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
        (; K, kmax, k, A) = spectral_stuff(setup; doaverage = false)
        ke = kinetic_energy(u, setup; interpolate_first = false)
        e = ke[Ip]
        e = fft(e)[ntuple(α -> 1:K[α], D)...]
        e = abs.(e) ./ prod(size(e))
        # nn = sqrt(T(prod(size(e))))
        # nn = T(size(e, 1))^T(3.1)
        # e = abs.(e) ./ nn
        e = A * reshape(e, :)
        e = max.(e, eps(T)) # Avoid log(0)
        ehat = Array(e)
        (; kmax, ehat)
    end
    (; kmax) = specs[1]
    # Build inertial slope above energy
    if D == 2
        # krange = [T(kmax)^(T(1) / 2), T(kmax)]
        krange = [T(50), T(400)]
    elseif D == 3
        krange = [T(kmax)^(T(2) / 3), T(kmax)]
    end
    slope, slopelabel =
        D == 2 ? (-T(3), L"$\kappa^{-3}") : (-T(5 / 3), L"$\kappa^{-5/3}")
    # slope, slopelabel = D == 2 ? (-T(3), "|k|⁻³") : (-T(5 / 3), "|k|⁻⁵³")
    # slope, slopelabel = D == 2 ? (-T(3), "κ⁻³") : (-T(5 / 3), "κ⁻⁵³")
    slopeconst = maximum(specs[1].ehat ./ (1:kmax) .^ slope)
    inertia = 2 .* slopeconst .* krange .^ slope
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
        ylabel = "e(κ)",
        xscale = log10,
        yscale = log10,
        limits = (1, kmax, T(1e-8), T(1)),
        title = "Kinetic energy",
    )
    # plotparts(i) = 1:specs[i].kmax, specs[i].ehat
    plotparts(i) = 1:specs[i].kmax+1, [specs[i].ehat; eps(T)]
    lines!(ax, 1:specs[1].kmax, specs[1].ehat; color = Cycled(1), label = "DNS")
    lines!(ax, 1:specs[2].kmax, specs[2].ehat; color = Cycled(4), label = "DNS, t = 0")
    lines!(ax, plotparts(3)...; color = Cycled(2), label = "Face average")
    lines!(ax, plotparts(4)...; color = Cycled(2))
    lines!(ax, plotparts(5)...; color = Cycled(2))
    lines!(ax, plotparts(6)...; color = Cycled(3), label = "Volume average")
    lines!(ax, plotparts(7)...; color = Cycled(3))
    lines!(ax, plotparts(8)...; color = Cycled(3))
    lines!(ax, krange, inertia; color = Cycled(1), label = slopelabel, linestyle = :dash)
    # axislegend(ax; position = :lb)
    autolimits!(ax)
    if D == 2
        limits!(ax, (T(0.68), T(520)), (T(1e-3), T(3e1)))
    elseif D == 3
        limits!(ax, ax.xaxis.attributes.limits[], (T(1e-1), T(3e1)))
    end
    fig
end
GC.gc()
CUDA.reclaim()

save("$output/priorspectra_$(D)D.pdf", fig)
