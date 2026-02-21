# Turbulent channel flow

println("Loading packages")
flush(stdout) # Prevent logging delay on Snellius

using AcceleratedKernels: AcceleratedKernels as AK
using Adapt
using CairoMakie
using CUDA
using CUDSS
using DelimitedFiles
using IncompressibleNavierStokes
using IncompressibleNavierStokes: IncompressibleNavierStokes as NS
using JLD2
using LaTeXStrings
# using WGLMakie
using WriteVTK

"Get turbulent channel flow problem setup."
function getproblem()

    println("Creating problem setup")
    flush(stdout) # Prevent logging delay on Snellius

    # Domain
    d = 1.0
    xlims = 0.0, 4 * π * d
    ylims = 0.0, 2 * d
    zlims = 0.0, 4 / 3 * π * d

    # Fluid
    viscosity = 1 / 180
    Re_tau = 180.0
    Re_m = 2800.0

    # Grid
    # n = 16
    # n = 32
    # n = 64
    # n = 128
    n = 256
    # n = 512

    nx, ny, nz = 2 * n, n, n
    # nx, ny, nz = 2 * n, 2 * n, n
    # nx, ny, nz = n, 2 * n, n
    
    stretch = 1.4

    setup = Setup(;
        boundary_conditions = (;
            u = (
                (PeriodicBC(), PeriodicBC()),
                (DirichletBC(), DirichletBC()),
                (PeriodicBC(), PeriodicBC()),
            ),
        ),
        x = (
            range(xlims..., nx + 1),
            tanh_grid(ylims..., ny + 1, stretch),
            range(zlims..., nz + 1),
        ),
        # x = (
        #     range(xlims..., nx + 1),
        #     range(ylims..., ny + 1),
        #     range(zlims..., nz + 1),
        # ),
        backend = CUDABackend(),
    )

    psolver = default_psolver(setup)

    Re_ratio = Re_m / Re_tau

    ustart = let
        Lx = xlims[2] - xlims[1]
        Ly = ylims[2] - ylims[1]
        Lz = zlims[2] - zlims[1]
        C = 9 / 8 * Re_ratio
        E = 1 / 10 * Re_ratio # 10% of average mean velocity
        function U(dim, x, y, z)
            ux =
                C * (1 - (y - Ly / 2)^8) +
                E * Lx / 2 * sinpi(y) * cospi(4 * x / Lx) * sinpi(2 * z / Lz)
            uy = -E * (1 - cospi(y)) * sinpi(4 * x / Lx) * sinpi(2 * z / Lz)
            uz = -E * Lz / 2 * sinpi(4 * x / Lx) * sinpi(y) * cospi(2 * z / Lz)
            return (dim == 1) * ux + (dim == 2) * uy + (dim == 3) * uz
        end
        velocityfield(setup, U; psolver)
    end

    return (; setup, psolver, ustart, viscosity)
end

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(
        f, state, t;
        setup, cache, viscosity,
    )
    navierstokes!(f, state, t; setup, cache, viscosity)
    @. f.u[:, :, :, 1] += 1 # Force is 1 in direction x
    # NS.wale_closure!(f.u, u, wale, cache.wale, setup)
    # NS.smagorinsky_closure!(f.u, u, smag, cache.smag, setup)
    return nothing
end

# Tell NS how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
function NS.get_cache(::typeof(force!), setup)
    # f(dim, x, y, z) = (dim == 1) * one(x)
    # bodyforce = velocityfield(setup, f; psolver, doproject = false)
    # (; bodyforce)
    # wale = NS.get_cache(NS.wale_closure!, setup)
    # smag = NS.get_cache(NS.smagorinsky_closure!, setup)
    # (; bodyforce, wale)
    # (; wale)
    # (; smag)
    return nothing
end

function compute_statistics!(statistics, u, setup)
    (; Iu, N) = setup
    nspace = (N[1] - 2) * (N[3] - 2) # Number of averaging volumes
    ubar, vbar, wbar = ntuple(3) do i
        sum(view(u, 2:(N[1] - 1), :, 2:(N[3] - 1), i); dims = (1, 3)) / nspace
    end
    statistics.ubar .+= view(ubar, 2:(N[2] - 1))
    AK.foreachindex(statistics.yplus) do jinner
        j = jinner + 1 # Account for ghost cell offset
        for k in 2:(N[3] - 1), i in 2:(N[1] - 1)
            # Face centered velocity fluctuations
            up = u[i, j, k, 1] - ubar[j]

            # Fluctuations interpolated to cell centers. For v, we subtract mean, then interpolate
            upc = (u[i, j, k, 1] + u[i - 1, j, k, 1]) / 2 - ubar[j]
            vpc = ((u[i, j, k, 2] - vbar[j]) + (u[i, j - 1, k, 2] - vbar[j - 1])) / 2
            wpc = (u[i, j, k, 3] + u[i, j, k - 1, 3]) / 2 - wbar[j]

            # Add to existing sums
            statistics.up_up[jinner] += up^2 / nspace
            statistics.up_up_up[jinner] += up^3 / nspace
            statistics.up_up_up_up[jinner] += up^4 / nspace

            # These are with cell-centered interpolations for collocation between u, v, w
            statistics.up_up_vp[jinner] += upc^2 * vpc / nspace
            statistics.up_wp[jinner] += upc * wpc / nspace
        end
    end
    return nothing
end

"Solve channel flow and average statistics over time."
function solve_statistics(setup, psolver, ustart, force!, params)

    println("Solve DNS and compute statistics.")
    flush(stdout) # Prevent logging delay on Snellius

    tstart = 0.0
    twarm = 10.0
    taverage = 20.0

    # Quantities from Vreman and Kuerten (2014):
    # y+ (=180*yc; yc is the (cell-central) y-location of u, w and p)
    # <u>
    # rms(u)  (=sqrt<u'u'>)
    # <u'u'u'>
    # <u'u'u'u'>
    # <u'u'v'>
    # <u'w'>
    #
    # The average is over x, z, and t

    yplus = setup.xu[1][2][2:(end - 1)] * 180
    statistics = (;
        yplus,
        ubar = zero(yplus),
        up_up = zero(yplus),
        up_up_up = zero(yplus),
        up_up_up_up = zero(yplus),
        up_up_vp = zero(yplus),
        up_wp = zero(yplus),
    )

    # Allocate buffers
    up = similar(ustart) # Fluctuating part of u
    temp = similar(ustart) # Storage for moments before averaging

    state = (; u = ustart)

    method = NS.LMWray3(; T = Float64)
    Δt = nothing
    Δt_min = nothing
    cfl = 0.45
    n_adapt_Δt = 1
    nupdate = 10
    ode_cache = NS.get_cache(method, state, setup)
    force_cache = NS.get_cache(force!, setup)
    stepper = create_stepper(method; setup, psolver, state, t = tstart)

    # Warm-up
    while stepper.t < twarm
        if stepper.n % n_adapt_Δt == 0
            # Change timestep based on operators
            Δt = cfl * NS.propose_timestep(force!, stepper.state, setup, params)
            Δt = isnothing(Δt_min) ? Δt : max(Δt, Δt_min)
        end

        # Make sure not to step past `t_end`
        Δt = min(Δt, twarm - stepper.t)

        if Δt < 1.0e-10 || Δt > 100 || isnan(Δt)
            @warn "Proposed time step $Δt is out of bounds. Stopping simulation."
            break
        end

        # Perform a single time step with the time integration method
        stepper = NS.timestep!(method, force!, stepper, Δt; params, ode_cache, force_cache)

        stepper.n % nupdate == 0 && println("Warm-up, t = $(round(stepper.t, sigdigits = 5)) / $twarm")
        flush(stdout) # Prevent logging delay on Snellius
    end

    # Register statistics
    nstep = 0
    while stepper.t < twarm + taverage
        if stepper.n % n_adapt_Δt == 0
            # Change timestep based on operators
            Δt = cfl * NS.propose_timestep(force!, stepper.state, setup, params)
            Δt = isnothing(Δt_min) ? Δt : max(Δt, Δt_min)
        end

        # Make sure not to step past `t_end`
        Δt = min(Δt, twarm + taverage - stepper.t)

        if Δt < 1.0e-10 || Δt > 100 || isnan(Δt)
            @warn "Proposed time step $Δt is out of bounds. Stopping simulation."
            break
        end

        # Perform a single time step with the time integration method
        stepper = NS.timestep!(method, force!, stepper, Δt; params, ode_cache, force_cache)
        nstep += 1

        compute_statistics!(statistics, stepper.state.u, setup)
        stepper.n % nupdate == 0 && println("Computing statistics, t = $(round(stepper.t - twarm, sigdigits = 5)) / $taverage")
        flush(stdout) # Prevent logging delay on Snellius
    end

    # Extract half profile for comparison with Vreman and Kuerten (2014)
    n = div(length(statistics.yplus), 2)
    statistics = map(s -> s[1:n], statistics)
    statistics = (; statistics..., label = "IncompressibleNavierStokes.jl")
    for symbol in [:ubar, :up_up, :up_up_up, :up_up_up_up, :up_up_vp, :up_wp]
        statistics[symbol] ./= nstep # time averaging factor
    end

    statistics = (; statistics..., urms = sqrt.(statistics.up_up))

    return adapt(Array, statistics)
end

"Extract data from Vreman."
function vremanstatistics()
    file = "Chan180_FD2_all/Chan180_FD2_basic_u.txt"
    fullfile = joinpath(@__DIR__, file)
    data = readdlm(fullfile, comments = true, comment_char = '%')
    statistics = (;
        yplus = data[:, 1],
        ubar = data[:, 2],
        urms = data[:, 3],
        up_up_up = data[:, 4],
        up_up_up_up = data[:, 5],
        up_up_vp = data[:, 6],
        up_wp = data[:, 7],
        label = "Vreman and Kuerten (2014)",
    )
    return statistics
end

"Make wall profile plot."
function plot_wall_profile(stats)
    for (key, title) in [
            :ubar => L"\langle u \rangle",
            :urms => L"\sqrt{\langle u'u' \rangle}",
            :up_up_up => L"\langle u'u'u' \rangle",
            :up_up_up_up => L"\langle u'u'u'u' \rangle",
            :up_up_vp => L"\langle u'u'v' \rangle",
            :up_wp => L"\langle u'w' \rangle",
        ]
        fig = Figure()
        ax = Axis(
            fig[1, 1];
            xlabel = L"y^{+}",
            ylabel = title,
            xscale = log10,
        )
        for s in stats
            scatter!(s.yplus, s[key]; s.label)
        end
        Legend(fig[1, 2], ax)
        path = "~/Projects/Thesis/Figures/Software" |> expanduser
        save(
            "$path/wallplot-$(key).pdf", fig; backend = CairoMakie,
            size = (700, 400)
        )
    end
    return
end

function statfile()
    path = joinpath(@__DIR__, "output", "ChannelVreman") |> mkpath
    return "$path/statistics.jld2"
end

if true # PROGRAM_FILE == @__FILE__
    psolver = nothing
    (; setup, psolver, ustart, viscosity) = getproblem()
    statistics = solve_statistics(setup, psolver, ustart, force!, (; viscosity))
    println("Saving statistics")
    flush(stdout) # Prevent logging delay on Snellius
    save_object(statfile(), statistics)
    statistics = load_object(statfile())
    statistics_ref = vremanstatistics()
    plot_wall_profile([statistics_ref, statistics])
end
