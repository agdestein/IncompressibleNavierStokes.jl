# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# using GLMakie
using CairoMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes: apply_bc_u!, total_kinetic_energy, diffusion!

# Output directory
output = "output/energy"
mkdir(output)

# Array type
ArrayType = Array
# using CUDA; ArrayType = CuArray;
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

clean() = GC.gc()

using CUDA;
ArrayType = CuArray;
CUDA.allowscalar(false);
clean() = (GC.gc(); CUDA.reclaim())

set_theme!()
set_theme!(; GLMakie = (; scalefactor = 1.5))

# 2D
T = Float64;
Re = T(10_000)
# ndns = 4096
ndns = 1024
# nles = 128
# ndns = 256
# nles = 128
nles = 64
compression = ndns ÷ nles
D = 2
kp = 20
lims = T(0), T(1);

dns, les = map((ndns, nles)) do n
    setup = Setup(ntuple(α -> LinRange(lims..., n + 1), D)...; Re, ArrayType)
    psolver = SpectralPressureSolver(setup)
    (; setup, psolver)
end;

# Create random initial conditions
u₀ = random_field(dns.setup, T(0); kp, dns.psolver);
clean()

state = (; u = u₀, t = T(0));

observe_u(dns, les, compression; Δt, nupdate = 1) =
    processor() do state
        u = state[].u
        v = zero.(FaceAverage()(u, les.setup, compression))
        # u = copy.(u₀)
        diffu = zero.(u)
        diffv = zero.(v)
        results = (;
            t = zeros(T, 0),
            Ku = zeros(T, 0),
            Kv = zeros(T, 0),
            Kuref = zeros(T, 0),
            Kvref = zeros(T, 0),
        )
        on(state) do (; u, t, n)
            n % nupdate == 0 || return
            apply_bc_u!(u, t, dns.setup)
            FaceAverage()(v, u, les.setup, compression)
            apply_bc_u!(v, t, les.setup)
            Ku = total_kinetic_energy(u, dns.setup)
            Kv = total_kinetic_energy(v, les.setup)
            push!(results.t, t)
            push!(results.Ku, Ku)
            push!(results.Kv, Kv)
            if n == 0
                push!(results.Kuref, Ku)
                push!(results.Kvref, Kv)
            else
                fill!.(diffu, 0)
                fill!.(diffv, 0)
                diffusion!(diffu, u, dns.setup)
                diffusion!(diffv, v, les.setup)
                apply_bc_u!(diffu, t, dns.setup)
                apply_bc_u!(diffv, t, les.setup)
                for α = 1:D
                    diffu[α] .*= u[α]
                    diffv[α] .*= v[α]
                end
                sum(sum.(diffu)) / ndns^D
                push!(
                    results.Kuref,
                    results.Kuref[end] + nupdate * Δt * sum(sum.(diffu)) / ndns^D,
                )
                push!(
                    results.Kvref,
                    results.Kvref[end] + nupdate * Δt * sum(sum.(diffv)) / nles^D,
                )
            end
        end
        state[] = state[] # Save initial conditions
        results
    end

# Δt = 5e-5
Δt = 1e-4

# Solve unsteady problem
@time state, outputs = solve_unsteady(
    dns.setup,
    u₀,
    # state.u,
    (T(0), T(5e-1));
    Δt,
    docopy = true,
    dns.psolver,
    processors = (
        rtp = realtimeplotter(; dns.setup, displayupdates = true, nupdate = 50),
        obs = observe_u(dns, les, compression; Δt, nupdate = 1),
        log = timelogger(; nupdate = 1),
    ),
);
clean()

outputs.obs.t
outputs.obs.Ku
outputs.obs.Kuref
outputs.obs.Kv
outputs.obs.Kvref

using CairoMakie

fig = with_theme() do
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "t", title = "Kinetic energy")
    lines!(
        ax,
        outputs.obs.t,
        outputs.obs.Ku;
        color = Cycled(1),
        linestyle = :solid,
        label = "DNS",
    )
    lines!(
        ax,
        outputs.obs.t,
        outputs.obs.Kuref;
        color = Cycled(1),
        linestyle = :dash,
        label = "DNS (reference)",
    )
    lines!(
        ax,
        outputs.obs.t,
        outputs.obs.Kv;
        color = Cycled(2),
        linestyle = :solid,
        label = "Filtered DNS",
    )
    lines!(
        ax,
        outputs.obs.t,
        outputs.obs.Kvref;
        color = Cycled(2),
        linestyle = :dash,
        label = "Filtered DNS (reference)",
    )
    axislegend(ax)
    fig
end

save("$output/energy.pdf", fig)

fig = with_theme() do
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(
        ax,
        outputs.obs.t,
        outputs.obs.Kv ./ outputs.obs.Ku;
        color = Cycled(2),
        linestyle = :solid,
        label = "Filtered DNS",
    )
    lines!(
        ax,
        outputs.obs.t,
        outputs.obs.Kvref ./ outputs.obs.Ku;
        color = Cycled(2),
        linestyle = :dash,
        label = "Filtered DNS (theoretical)",
    )
    # axislegend(ax)
    fig
end
