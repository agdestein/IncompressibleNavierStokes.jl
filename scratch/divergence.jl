# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

#md using CairoMakie
using GLMakie
using IncompressibleNavierStokes
using IncompressibleNavierStokes: momentum!, divergence!, project!, apply_bc_u!
using LinearAlgebra
using Printf

# Output directory
output = "output/DecayingTurbulence2D"

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

function observe_v(dnsobs, Φ, les, compression, psolver)
    (; ArrayType, grid) = les
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
        (; dimension, x) = dns.grid
        T = eltype(x[1])
        D = dimension()
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

# A 2D grid is a Cartesian product of two vectors
ndns = 1024
lims = T(0), T(1)
D = 2
dns = Setup(ntuple(α -> LinRange(lims..., ndns + 1), D)...; Re, ArrayType);

filters = map([
    (FaceAverage(), 64),
    (FaceAverage(), 128),
    (FaceAverage(), 256),
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
u₀ = random_field(dns, T(0); psolver = psolver_dns);

# Solve unsteady problem
state, outputs = solve_unsteady(
    dns,
    u₀,
    (T(0), T(0.01));
    Δt = T(1e-4),
    psolver = psolver_dns,
    processors = (
        rtp = realtimeplotter(; setup = dns, nupdate = 1),
        obs = observe_u(dns, psolver_dns, filters),
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
        log = timelogger(; nupdate = 100),
    ),
);
(; u, t) = state;

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
