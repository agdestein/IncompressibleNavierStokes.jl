# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

using GLMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes: apply_bc_u!, total_kinetic_energy, diffusion!

# Output directory
output = "output/energy"

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
Re = T(4_000)
# ndns = 1024
# nles = 128
ndns = 256
nles = 32
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
            push!(results.t, t)
            push!(results.Ku, Ku)
            push!(results.Kv, Kv)
        end
        state[] = state[] # Save initial conditions
        results
    end

Δt = 1e-4

# Solve unsteady problem
@time state, outputs = solve_unsteady(
    dns.setup,
    u₀,
    # state.u,
    (T(0), T(1e-1));
    Δt,
    docopy = true,
    dns.psolver,
    processors = (
        # rtp = realtimeplotter(; dns.setup, nupdate = 5),
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

with_theme() do
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, outputs.obs.t, outputs.obs.Ku; label = "Ku")
    lines!(ax, outputs.obs.t, outputs.obs.Kuref; label = "Kuref")
    lines!(ax, outputs.obs.t, outputs.obs.Kv; label = "Kv")
    lines!(ax, outputs.obs.t, outputs.obs.Kvref; label = "Kvref")
    axislegend()
    fig
end
