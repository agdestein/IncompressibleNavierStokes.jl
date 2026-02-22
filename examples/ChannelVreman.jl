# Turbulent channel flow

println("Loading packages")
flush(stdout) # Prevent logging delay on Snellius

using Adapt
using CairoMakie
using CUDA
using CUDSS
using DelimitedFiles
using JLD2
using ProgressLogging
using Printf
using LaTeXStrings
# using WGLMakie
using WriteVTK

import AcceleratedKernels as AK
import IncompressibleNavierStokes as NS

"Get turbulent channel flow problem parameters."
function getproblem()
    # Domain
    H = 1.0 # Channel half width
    Lx = 4 * π * H
    Ly = 2 * H
    Lz = 4 / 3 * π * H

    # Flow
    u_tau = 1.0
    Re_tau = 180.0
    Re_m = 2800.0
    viscosity = u_tau * H / Re_tau

    # Grid
    n = 16
    # n = 32
    # n = 64
    # n = 128
    # n = 256
    # n = 512

    # Number of volumes in each dimension
    # nx, ny, nz = 2 * n, 4 * n, n
    # nx, ny, nz = 2 * n, n, n
    nx, ny, nz = 2 * n, 2 * n, n
    # nx, ny, nz = n, 2 * n, n

    stretch = nothing # Uniform wall-normal grid
    # stretch = 3.0 # Stretched wall-normal grid

    # Simulation time
    twarmup = 1.0 * H / u_tau # Warm-up time
    taverage = 2.0 * H / u_tau # Averaging time for statistics
    tsimulation = twarmup + taverage

    # Time stepping
    cfl = 0.8 # Time step control
    nsave = 20 # Number of times to save statistics

    return (;
        H, Lx, Ly, Lz,
        u_tau, Re_tau, Re_m, viscosity,
        nx, ny, nz, stretch,
        twarmup, taverage, tsimulation,
        cfl, nsave,
    )
end

"Get setup, psolver, and initial conditions."
function getsetup(problem)
    isuniform = isnothing(problem.stretch)

    println("Creating problem setup")
    flush(stdout) # Prevent logging delay on Snellius

    ax_x = range(0.0, problem.Lx, problem.nx + 1)
    ax_y = if isuniform
        range(0.0, problem.Ly, problem.ny + 1)
    else
        NS.tanh_grid(0.0, problem.Ly, problem.ny + 1, problem.stretch)
    end
    ax_z = range(0.0, problem.Lz, problem.nz + 1)

    setup = NS.Setup(;
        boundary_conditions = (;
            u = (
                (NS.PeriodicBC(), NS.PeriodicBC()),
                (NS.DirichletBC(), NS.DirichletBC()),
                (NS.PeriodicBC(), NS.PeriodicBC()),
            ),
        ),
        x = (ax_x, ax_y, ax_z),
        backend = CUDABackend(),
    )

    psolver = if isuniform
        NS.psolver_transform(setup) # FFT-based
    else
        NS.default_direct(setup) # Matrix decomposition
        # psolver_cg(setup; abstol = 1e-6)
        # psolver_cg_matrix(setup; abstol = 1e-4)
    end

    # Initial conditions
    (; Lx, Ly, Lz, Re_m, Re_tau) = problem
    Re_ratio = Re_m / Re_tau
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
    ustart = NS.velocityfield(setup, U; psolver)

    return (; setup, psolver, ustart)
end

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(
        f, state, t;
        setup, cache, viscosity,
    )
    NS.navierstokes!(f, state, t; setup, cache, viscosity)
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

function compute_statistics!(buffers, uvw, setup)
    (; N) = setup
    (;
        ubar, vbar, wbar,
        up_up, vp_vp, wp_wp,
        up_up_up, up_up_up_up,
        up_up_vp, up_wp,
    ) = buffers
    nspace = (N[1] - 2) * (N[3] - 2) # Number of averaging volumes

    # First compute ubar, vbar, wbar
    AK.foraxes(uvw, 2) do j
        ubar_j = 0.0
        vbar_j = 0.0
        wbar_j = 0.0
        for k in 2:(N[3] - 1), i in 2:(N[1] - 1)
            ubar_j += uvw[i, j, k, 1]
            vbar_j += uvw[i, j, k, 2]
            wbar_j += uvw[i, j, k, 3]
        end
        ubar[j] = ubar_j / nspace
        vbar[j] = vbar_j / nspace
        wbar[j] = wbar_j / nspace
    end

    # Loop over wall-normal planes
    AK.foraxes(uvw, 2) do j

        # Initialize sums
        up_up_j = 0.0
        vp_vp_j = 0.0
        wp_wp_j = 0.0
        up_up_up_j = 0.0
        up_up_up_up_j = 0.0
        up_up_vp_j = 0.0
        up_wp_j = 0.0

        # Sum over current wall-normal plane
        for k in 2:(N[3] - 1), i in 2:(N[1] - 1)
            # Face centered velocity fluctuations
            up = uvw[i, j, k, 1] - ubar[j]
            vp = uvw[i, j, k, 2] - vbar[j]
            wp = uvw[i, j, k, 3] - wbar[j]

            # For j = 1, there is no left volume,
            # but the volume is flat so we can just use jleft = j
            jleft = max(j - 1, 1)

            # Fluctuations interpolated to cell centers. For v, we subtract mean, then interpolate
            upc = (uvw[i, j, k, 1] + uvw[i - 1, j, k, 1]) / 2 - ubar[j]
            vpc = ((uvw[i, j, k, 2] - vbar[j]) + (uvw[i, jleft, k, 2] - vbar[jleft])) / 2
            wpc = (uvw[i, j, k, 3] + uvw[i, j, k - 1, 3]) / 2 - wbar[j]

            # Add to existing sums
            up_up_j += up^2
            vp_vp_j += vp^2
            wp_wp_j += wp^2
            up_up_up_j += up^3
            up_up_up_up_j += up^4

            # These are with cell-centered interpolations for collocation between u, v, w
            up_up_vp_j += upc^2 * vpc
            up_wp_j += upc * wpc
        end

        # Write means
        up_up[j] = up_up_j / nspace
        vp_vp[j] = vp_vp_j / nspace
        wp_wp[j] = wp_wp_j / nspace
        up_up_up[j] = up_up_up_j / nspace
        up_up_up_up[j] = up_up_up_up_j / nspace
        up_up_vp[j] = up_up_vp_j / nspace
        up_wp[j] = up_wp_j / nspace
    end

    return nothing
end

"Solve channel flow and average statistics over time."
function solve(setup, psolver, ustart, force!, params, problem)

    @info "Solving DNS and computing statistics."
    flush(stderr) # Prevent logging delay on Snellius

    (; tsimulation, cfl, nsave) = problem
    (; Δ) = setup

    tstart = 0.0
    Δt = 0.0 # Will be overridden by adaptive time stepping
    Δt_save = tsimulation / nsave

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

    buffers = (;
        ubar = zero(Δ[2]),
        vbar = zero(Δ[2]),
        wbar = zero(Δ[2]),
        up_up = zero(Δ[2]),
        vp_vp = zero(Δ[2]),
        wp_wp = zero(Δ[2]),
        up_up_up = zero(Δ[2]),
        up_up_up_up = zero(Δ[2]),
        up_up_vp = zero(Δ[2]),
        up_wp = zero(Δ[2]),
    )

    statistics = fill(map(Array, buffers), 0)
    times = zeros(0)

    state = (; u = ustart)

    method = NS.LMWray3(; T = Float64)
    ode_cache = NS.get_cache(method, state, setup)
    force_cache = NS.get_cache(force!, setup)
    stepper = NS.create_stepper(method; setup, psolver, state, t = tstart)

    # Step through all the save points
    @progress for isave in 0:nsave
        # Step until next save point.
        # For the step, the while loop is never entered.
        tstop = isave * Δt_save
        while stepper.t < prevfloat(tstop)
            # Change timestep based on operators
            Δt = cfl * NS.propose_timestep(force!, stepper.state, setup, params)

            # Make sure not to step past `tstop`
            Δt = min(Δt, tstop - stepper.t)

            if Δt < 1.0e-10 || Δt > 100 || isnan(Δt)
                @warn "Proposed time step $Δt is out of bounds. Stopping simulation."
                break
            end

            # Perform a single time step with the time integration method
            stepper = NS.timestep!(method, force!, stepper, Δt; params, ode_cache, force_cache)
        end

        # Compute statistics, copy to CPU, and store in list
        compute_statistics!(buffers, stepper.state.u, setup)
        push!(statistics, map(Array, buffers))
        push!(times, stepper.t)

        flush(stdout) # Prevent logging delay on Snellius
        flush(stderr) # Prevent logging delay on Snellius
    end

    # Extract half profile for comparison with Vreman and Kuerten (2014)
    statfile = joinpath(getoutdir(problem), "statistics.jld2")
    @info "Saving statistics to: $statfile"
    flush(stderr) # Prevent logging delay on Snellius
    save_object(statfile, statistics)

    return statistics
end

function process_statseries(statistics, setup, problem)
    (; tsimulation, twarmup, nsave) = problem
    # Exclude zero
    nhalf = div(problem.ny, 2)
    ycenter = setup.xp[2][2:(nhalf + 1)] * problem.u_tau / problem.viscosity
    yedge = setup.xu[2][2][2:(nhalf + 1)] * problem.u_tau / problem.viscosity

    # Filter out warm-up stats
    istart = findfirst(i -> tsimulation / nsave * i > twarmup, 0:(length(statistics) - 1))
    stats_use = statistics[istart:end]

    # Compute time averages
    averages = map(keys(statistics[1])) do key
        # Average over time
        statavg = sum(stats_use) do s
            getindex(s, key)[2:(nhalf + 1)]
        end / length(stats_use)
        key => statavg
    end

    return (; averages..., ycenter, yedge, label = "IncompressibleNavierStokes.jl")
end

"Extract data from Vreman."
function vremanstatistics()
    file = "Chan180_FD2_all/Chan180_FD2_basic_u.txt"
    fullfile = joinpath(@__DIR__, file)
    data = readdlm(fullfile, comments = true, comment_char = '%')
    statistics = (;
        ycenter = data[:, 1],
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
            scatter!(s.ycenter, s[key]; s.label)
        end
        Legend(fig[1, 2], ax)
        # path = "~/Projects/Thesis/Figures/Software" |> expanduser
        plotfile = joinpath(getoutdir(problem), "wallplot-$(key).pdf")
        println("Saving plot to: $plotfile")
        save(plotfile, fig; backend = CairoMakie, size = (700, 400))
    end
    return
end

function getoutdir(problem)
    (; nx, ny, nz, stretch) = problem
    name = "ChannelVreman-nx=$nx-ny=$ny-nz=$nz-stretch=$stretch"
    return joinpath(@__DIR__, "output", name) |> mkpath
end

function show_problem(setup, problem)
    (; Δ) = setup
    (; nx, ny, nz, u_tau, viscosity) = problem
    Δxp_max = maximum(Δ[1][2:(end - 1)]) * u_tau / viscosity
    Δyp_max = maximum(Δ[2][2:(end - 1)]) * u_tau / viscosity
    Δzp_max = maximum(Δ[3][2:(end - 1)]) * u_tau / viscosity
    Δxp_min = minimum(Δ[1][2:(end - 1)]) * u_tau / viscosity
    Δyp_min = minimum(Δ[2][2:(end - 1)]) * u_tau / viscosity
    Δzp_min = minimum(Δ[3][2:(end - 1)]) * u_tau / viscosity
    @printf("Max grid spacing in wall units: Δx+ = %.4g, Δy+ = %.4g, Δz+ = %.4g\n", Δxp_max, Δyp_max, Δzp_max)
    @printf("Min grid spacing in wall units: Δx+ = %.4g, Δy+ = %.4g, Δz+ = %.4g\n", Δxp_min, Δyp_min, Δzp_min)
    println("Grid size: nx = $nx, ny = $ny, nz = $nz")
    return nothing
end

setup.Iu

# Main script
problem = getproblem()
(; setup, psolver, ustart) = getsetup(problem)
show_problem(setup, problem)
statistics = solve(setup, psolver, ustart, force!, (; problem.viscosity), problem)
# statistics = load_object(statfile)
statistics_ins = process_statseries(statistics, setup, problem)
statistics_ref = vremanstatistics()
plot_wall_profile([statistics_ref, statistics])
