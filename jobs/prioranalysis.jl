# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

@info "Loading packages"

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
using JLD2
using LinearAlgebra
using Printf
using FFTW

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
        E = zeros(T, 0),
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

        Eu = sum(α -> sum(abs2, view(u[α], ntuple(b -> 2:size(u[α], b)-1, 1:D)), 1:D)
        Ev = norm_v^2
        E = compression * Ev / Eu
        push!(results.E, E)
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

# Output directory
output = "output/prioranalysis/dimension$D"
ispath(output) || mkpath(output)

# Array type
ArrayType = Array
# using CUDA; ArrayType = CuArray;
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using CUDA;
ArrayType = CuArray;
CUDA.allowscalar(false);

# 3D
T = Float32
# T = Float64
Re = T(2_000)
# ndns = 512
ndns = 1024
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

# # 2D 
# T = Float64;
# Re = T(10_000)
# ndns = 4096
# D = 2
# kp = 20
# Δt = T(5e-5)
# filterdefs = [
#     (FaceAverage(), 64),
#     (FaceAverage(), 128),
#     (FaceAverage(), 256),
#     (VolumeAverage(), 64),
#     (VolumeAverage(), 128),
#     (VolumeAverage(), 256),
# ]

# Setup
@info "Building setup"
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
@info "Generating initial conditions"
u₀ = random_field(dns, T(0); kp, psolver = psolver_dns);
GC.gc()
CUDA.reclaim()


# Solve unsteady problem
@info "Launching time stepping"
@time state, outputs = solve_unsteady(
    dns,
    u₀,
    (T(0), T(1e-1));
    Δt,
    docopy = true,
    psolver = psolver_dns,
    processors = (
        obs = observe_u(dns, psolver_dns, filters; nupdate = 20),
        log = timelogger(; nupdate = 20),
    ),
);
GC.gc()
CUDA.reclaim()

# Save final solution
@info "Saving final solution"
jldsave("$output/finalsolution.jld", u = Array.(state.u))

# u = load("$output/finalsolution.jld")["u"]
# state = (; u = ArrayType.(u), t = T(1e-1))

# fil = filters[2];
# apply_bc_u!(state.u, T(0), dns);
# v = fil.Φ(state.u, fil.setup, fil.compression);
# apply_bc_u!(v, T(0), fil.setup);
# Fv = momentum(v, T(0), fil.setup);
# apply_bc_u!(Fv, T(0), fil.setup);
# PFv = project(Fv, fil.setup; psolver = fil.psolver);
# apply_bc_u!(PFv, T(0), fil.setup);
# F = momentum(state.u, T(0), dns);
# apply_bc_u!(F, T(0), dns);
# PF = project(F, dns; psolver = psolver_dns);
# apply_bc_u!(PF, T(0), dns);
# ΦPF = fil.Φ(PF, fil.setup, fil.compression);
# apply_bc_u!(ΦPF, T(0), fil.setup);
# c = ΦPF .- PFv
# apply_bc_u!(c, T(0), fil.setup)

# with_theme(; fontsize = 25) do
#     fig = fieldplot(
#         (; u = u₀, t = T(0));
#         setup = dns,
#         # type = image,
#         # colormap = :viridis,
#         docolorbar = false,
#         size = (500, 500),
#         title = "u₀",
#     )
#     save("$output/ustart.png", fig)
#     fig = fieldplot(
#         state,
#         setup = dns,
#         # type = image,
#         # colormap = :viridis,
#         docolorbar = false,
#         size = (500, 500),
#         title = "u",
#     )
#     save("$output/u.png", fig)
#     fig = fieldplot(
#         (; u = v, t = T(0));
#         fil.setup,
#         # type = image,
#         # colormap = :viridis,
#         # fieldname = 1,
#         docolorbar = false,
#         size = (500, 500),
#         # title = "ubar"
#         title = "ū",
#     )
#     save("$output/v.png", fig)
#     fig = fieldplot(
#         (; u = PF, t = T(0));
#         setup = dns,
#         # type = image,
#         # colormap = :viridis,
#         # fieldname = 1,
#         docolorbar = false,
#         size = (500, 500),
#         title = "PF(u)",
#     )
#     save("$output/PFu.png", fig)
#     fig = fieldplot(
#         (; u = PFv, t = T(0));
#         fil.setup,
#         # type = image,
#         # colormap = :viridis,
#         # fieldname = 1,
#         docolorbar = false,
#         size = (500, 500),
#         # title = "PF(ubar)"
#         title = "P̄F̄(ū)",
#     )
#     save("$output/PFv.png", fig)
#     fig = fieldplot(
#         (; u = ΦPF, t = T(0));
#         fil.setup,
#         # type = image,
#         # colormap = :viridis,
#         # fieldname = 1,
#         docolorbar = false,
#         size = (500, 500),
#         title = "ΦPF(u)",
#     )
#     save("$output/PhiPFu.png", fig)
#     fig = fieldplot(
#         (; u = c, t = T(0));
#         fil.setup,
#         # type = image,
#         # colormap = :viridis,
#         # fieldname = 1,
#         # fieldname = :velocity,
#         docolorbar = false,
#         size = (500, 500),
#         # title = "c(u, ubar)"
#         title = "c(u, ū)",
#     )
#     save("$output/c.png", fig)
# end

# 3D fieldplot #################################################################

# fieldplot(
#     # (; u = u₀, t = T(0));
#     state;
#     setup = dns,
#     fieldname = :eig2field,
#     # levels = LinRange(T(2), T(10), 10),
#     levels = LinRange(T(4), T(12), 10),
#     # levels = LinRange(-1.0f0, 3.0f0, 5),
#     # levels = LinRange(-2.0f0, 2.0f0, 5),
#     # levels = 5,
#     docolorbar = false,
#     # size = (800, 800),
#     size = (600, 600),
# )
#
# fname = "$output/lambda2/Re$(Int(Re))_start.png"
# fname = "$output/lambda2/Re$(Int(Re))_end.png"
# save(fname, current_figure())
# run(`convert $fname -trim $fname`) # Requires imagemagick

# i = 3
# fieldplot(
#     (; u = filters[i].Φ(state.u, filters[i].setup, filters[i].compression), t = T(0));
#     setup = filters[i].setup,
#     fieldname = :eig2field,
#     # levels = LinRange(T(2), T(10), 10),
#     levels = LinRange(T(4), T(12), 10),
#     # levels = LinRange(-1.0f0, 3.0f0, 5),
#     # levels = LinRange(-2.0f0, 2.0f0, 5),
#     # levels = 5,
#     docolorbar = false,
#     # size = (800, 800),
#     size = (600, 600),
# )
#
# fname = "$output/lambda2/Re$(Int(Re))_end_filtered.png"
# save(fname, current_figure())
# run(`convert $fname -trim $fname`) # Requires imagemagick

@info "Computing statistics"
begin
    println("Φ\t\tM\tDu\tPv\tPc\tc\tE")
    for o in outputs.obs
        nt = length(o.t)
        Dv = sum(o.Dv) / nt
        Pc = sum(o.Pc) / nt
        Pv = sum(o.Pv) / nt
        c = sum(o.c) / nt
        E = sum(o.E) / nt
        @printf(
            "%s\t%d^%d\t%.2g\t%.2g\t%.2g\t%.2g\t%.2g\n",
            # "%s &\t\$%d^%d\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$ &\t\$%.2g\$\n",
            typeof(o.Φ),
            o.Mα,
            D,
            Dv,
            Pv,
            Pc,
            c,
            E,
        )
    end
end;

# To free up memory
psolver_dns = nothing
# fig = lines([1, 2, 3])
GC.gc()
CUDA.reclaim()

# Plot predicted spectra
@info "Computing and plotting spectra"
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

# save("$output/spectra_$(D)D_linear_Re$(Int(Re)).pdf", fig)
save("$output/spectra_$(D)D_dyadic_Re$(Int(Re)).pdf", fig)
