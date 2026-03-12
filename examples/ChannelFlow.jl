# # Turbulent channel flow
#
# Turbulent channel flow setup from 
# > A. W. Vreman and J. G. M. Kuerten.
# > “Comparison of Direct Numerical Simulation Databases of Turbulent Channel Flow at ``Re_\tau = 180``.”
# > In: Physics of Fluids 26.1 (Jan. 2014), p. 015102.
# > doi: [10.1063/1.4861064](https://doi.org/10.1063/1.4861064).
#
# This script contains two experiments:
# 1. DNS.
# 2. LES with various eddy-viscosity closures.
#
# All settings are toggled in the function `getproblem()`.

@info "Loading packages"
flush(stderr) # Prevent logging delay on Snellius

using Adapt
using CairoMakie
using CUDA
using CUDSS
using DelimitedFiles
using Downloads
using JLD2
using ProgressMeter
using Printf
using LaTeXStrings

import AcceleratedKernels as AK
import IncompressibleNavierStokes as NS

"Get turbulent channel flow problem parameters."
function getproblem()
    ## Domain
    H = 1.0 # Channel half width
    Lx = 4 * π * H
    Ly = 2 * H
    Lz = 4 / 3 * π * H

    ## Flow
    u_tau = 1.0
    Re_tau = 180.0
    Re_m = 2800.0
    viscosity = u_tau * H / Re_tau
    forcing = 1.0

    doles = true # DNS or LES
    dosimulation = false # Switch for rerunning simulations

    if doles
        n = 64
        nx, ny, nz = 2 * n, n, n
        stretch = 1.4 # Stretched wall-normal grid
    else
        ## Grid
        ## n = 16
        ## n = 32
        ## n = 64
        ## n = 128
        n = 256
        ## n = 512

        ## Number of volumes in each dimension
        ## nx, ny, nz = 2 * n, n, n
        ## nx, ny, nz = 2 * n, 2 * n, n
        nx, ny, nz = 2 * n, 4 * n, n

        stretch = nothing # Uniform wall-normal grid
        ## stretch = 1.4 # Stretched wall-normal grid
        ## stretch = 2.0 # Stretched wall-normal grid
    end

    ## Simulation time
    twarmup = 15.0 * H / u_tau # Warm-up time
    taverage = 10.0 * H / u_tau # Averaging time for statistics
    tsimulation = twarmup + taverage

    ## Time stepping
    cfl = 0.9 # Time step control
    nsave = 2500 # Number of times to save statistics

    ## Closure models
    C_smag = 0.1
    C_wale = 0.5
    C_qr = sqrt(3 / 2) / π
    C_vreman = sqrt(2.5 * C_smag^2)

    return (;
        dosimulation, doles,
        H, Lx, Ly, Lz,
        u_tau, Re_tau, Re_m,
        viscosity, forcing,
        nx, ny, nz, stretch,
        twarmup, taverage, tsimulation,
        cfl, nsave,
        C_smag, C_wale, C_qr, C_vreman,
    )
end

"Get setup, psolver, and initial conditions."
function getsetup(problem)
    (; Lx, Ly, Lz, nx, ny, nz, stretch) = problem
    isuniform = isnothing(stretch)

    @info "Creating problem setup"
    flush(stderr) # Prevent logging delay on Snellius

    ax_x = range(0.0, Lx, nx + 1)
    ax_y = if isuniform
        range(0.0, Ly, ny + 1)
    else
        NS.tanh_grid(0.0, Ly, ny, stretch)
    end
    ax_z = range(0.0, Lz, nz + 1)

    return NS.Setup(;
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
end

function getheavystuff(setup, problem)
    (; stretch, Lx, Ly, Lz, Re_m, Re_tau) = problem

    isuniform = isnothing(stretch)
    psolver = if isuniform
        NS.psolver_transform(setup) # FFT-based
    else
        NS.psolver_direct(setup) # Matrix decomposition
        ## psolver_cg(setup; abstol = 1e-6)
        ## psolver_cg_matrix(setup; abstol = 1e-4)
    end

    ## Initial conditions
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

    return (; psolver, ustart)
end

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force_nomo!(f, state, t; setup, cache, viscosity, forcing)
    NS.navierstokes!(f, state, t; setup, cache, viscosity)
    @. f.u[:, :, :, 1] += forcing # Forcing in direction x
    return nothing
end
function force_eddy!(f, state, t; setup, cache, viscosity, forcing, eddyviscosity)
    NS.navierstokes!(f, state, t; setup, cache, viscosity)
    NS.eddy_viscosity_closure!(eddyviscosity, f.u, state.u, cache, setup)
    @. f.u[:, :, :, 1] += forcing # Forcing in direction x
    return nothing
end

# Tell NS how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
NS.get_cache(::typeof(force_nomo!), setup) = nothing
NS.get_cache(::typeof(force_eddy!), setup) = NS.get_cache(NS.eddy_viscosity_closure!, setup)

"Compute spatial statistics for a single velocity snapshot."
function compute_statistics!(buffers, uvw, setup, cache)
    dovisc = !isnothing(cache)
    (; N) = setup
    (;
        ubar, vbar, wbar,
        up_up, vp_vp, wp_wp,
        up_up_up, up_up_up_up,
        up_up_vp, up_wp,
        visc,
    ) = buffers
    nspace = (N[1] - 2) * (N[3] - 2) # Number of averaging volumes

    ## First compute ubar, vbar, wbar
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

    ## Average eddy viscosity
    dovisc && AK.foraxes(uvw, 2) do j
        visc_j = 0.0
        for k in 2:(N[3] - 1), i in 2:(N[1] - 1)
            visc_j += dovisc ? cache.visc[i, j, k] : 0.0
        end
        visc[j] = visc_j / nspace
    end

    ## Loop over wall-normal planes
    AK.foraxes(uvw, 2) do j

        ## Initialize sums
        up_up_j = 0.0
        vp_vp_j = 0.0
        wp_wp_j = 0.0
        up_up_up_j = 0.0
        up_up_up_up_j = 0.0
        up_up_vp_j = 0.0
        up_wp_j = 0.0

        ## Sum over current wall-normal plane
        for k in 2:(N[3] - 1), i in 2:(N[1] - 1)
            ## Face centered velocity fluctuations
            up = uvw[i, j, k, 1] - ubar[j]
            vp = uvw[i, j, k, 2] - vbar[j]
            wp = uvw[i, j, k, 3] - wbar[j]

            ## For j = 1, there is no left volume,
            ## but the volume is flat so we can just use jleft = j
            jleft = max(j - 1, 1)

            ## Fluctuations interpolated to cell centers. For v, we subtract mean, then interpolate
            upc = (uvw[i, j, k, 1] + uvw[i - 1, j, k, 1]) / 2 - ubar[j]
            vpc = ((uvw[i, j, k, 2] - vbar[j]) + (uvw[i, jleft, k, 2] - vbar[jleft])) / 2
            wpc = (uvw[i, j, k, 3] + uvw[i, j, k - 1, 3]) / 2 - wbar[j]

            ## Add to existing sums
            up_up_j += up^2
            vp_vp_j += vp^2
            wp_wp_j += wp^2
            up_up_up_j += up^3
            up_up_up_up_j += up^4

            ## These are with cell-centered interpolations for collocation between u, v, w
            up_up_vp_j += upc^2 * vpc
            up_wp_j += upc * wpc
        end

        ## Write means
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
function solve(setup, psolver, ustart, force!, params, problem, filename, desc)

    @info "Solving with \"$desc\" and computing statistics."
    flush(stderr) # Prevent logging delay on Snellius

    (; tsimulation, cfl, nsave) = problem
    (; Δ) = setup

    tstart = 0.0
    Δt = 0.0 # Will be overridden by adaptive time stepping
    Δt_save = tsimulation / nsave

    ## Quantities from Vreman and Kuerten (2014):
    ## y+ (=180*yc; yc is the (cell-central) y-location of u, w and p)
    ## <u>
    ## rms(u)  (=sqrt<u'u'>)
    ## <u'u'u'>
    ## <u'u'u'u'>
    ## <u'u'v'>
    ## <u'w'>
    ##
    ## The average is over x, z, and t

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
        visc = zero(Δ[2])
    )

    statistics = fill(map(Array, buffers), 0)
    times = zeros(0)

    state = (; u = ustart)

    method = NS.LMWray3(; T = Float64)
    ode_cache = NS.get_cache(method, state, setup)
    force_cache = NS.get_cache(force!, setup)
    stepper = NS.create_stepper(method; setup, psolver, state, t = tstart)

    ## Step through all the save points
    everythingfine = true
    prog = Progress(nsave + 1; desc)
    for isave in 0:nsave
        ## Step until next save point.
        ## For the first step, the while loop is never entered.
        tstop = isave * Δt_save
        isubstep = 0
        while stepper.t < prevfloat(tstop)
            ## Change timestep based on operators
            Δt = cfl * NS.propose_timestep(force!, stepper.state, setup, params)

            ## Make sure not to step past `tstop`
            Δt = min(Δt, tstop - stepper.t)

            if Δt < 1.0e-10 || Δt > 100 || isnan(Δt)
                @warn "Proposed time step $Δt is out of bounds. Stopping simulation."
                everythingfine = false
                break
            end

            ## Perform a single time step with the time integration method
            stepper = NS.timestep!(method, force!, stepper, Δt; params, ode_cache, force_cache)
            isubstep += 1
        end

        ## Compute statistics, copy to CPU, and store in list
        compute_statistics!(buffers, stepper.state.u, setup, force_cache)
        push!(statistics, map(Array, buffers))
        push!(times, stepper.t)

        next!(
            prog; showvalues = [
                ("Δt", cfl * NS.propose_timestep(force!, stepper.state, setup, params)),
                ("substeps", isubstep),
                ("time", stepper.t),
            ]
        )
        flush(stdout) # Prevent logging delay on Snellius
        flush(stderr) # Prevent logging delay on Snellius

        everythingfine || break
    end

    ## Save statistics to disk
    statfile = joinpath(getoutdir(problem), filename)
    @info "Saving statistics to: $statfile"
    flush(stderr) # Prevent logging delay on Snellius
    save_object(statfile, statistics)

    return nothing
end

function process_statseries(statistics, setup, problem, label)
    (; u_tau, viscosity, tsimulation, twarmup, nsave, ny) = problem
    (; xp, xu) = setup

    ## Get yplus coordinates until half channel height.
    ## Exclude zero.
    nhalf = div(ny, 2)
    ycenter = xp[2][2:(nhalf + 1)] * u_tau / viscosity |> Array
    yedge = xu[2][2][2:(nhalf + 1)] * u_tau / viscosity |> Array

    ## Filter out warm-up stats
    istart = findfirst(i -> tsimulation / nsave * (i - 1) > twarmup, eachindex(statistics))
    stats_use = statistics[istart:end]

    ## Compute time averages
    averages = map(keys(statistics[1])) do key
        statavg = sum(stats_use) do s
            ## Average over the two symmetric values also
            if key == :vp_vp
                (getindex(s, key)[2:(nhalf + 1)] .+ getindex(s, key)[(end - 2):-1:(nhalf + 1)]) ./ 2
            elseif key == :vbar
                ## This quantity is signed, but the sign we want is "velocity away from the wall".
                ## Therefore flip the sign in the average.
                ## Note: This quantity is probably zero anyway.
                (getindex(s, key)[2:(nhalf + 1)] .- getindex(s, key)[(end - 2):-1:(nhalf + 1)]) ./ 2
            elseif key == :up_up_vp
                ## This should also have a flipped sign, since up^2 is positive,
                ## but vp is probably inverse accross midline.
                (getindex(s, key)[2:(nhalf + 1)] .- getindex(s, key)[(end - 1):-1:(nhalf + 2)]) ./ 2
            else
                (getindex(s, key)[2:(nhalf + 1)] .+ getindex(s, key)[(end - 1):-1:(nhalf + 2)]) ./ 2
            end
        end / length(stats_use)
        key => statavg
    end |> NamedTuple

    return (;
        averages...,
        urms = sqrt.(averages.up_up),
        vrms = sqrt.(averages.vp_vp),
        wrms = sqrt.(averages.wp_wp),
        ycenter, yedge,
        label,
    )
end

"Download data from Vreman."
function downloaddata()
    url = "http://www.vremanresearch.nl"
    path = joinpath(@__DIR__, "Chan180_FD2_all")
    if ispath(path)
        @info "Reference data already exists at $path. Skipping download."
    else
        @info "Downloading reference data from $url to $path."
        mkpath(path)
        for sym in ["u", "v", "w"]
            file = "Chan180_FD2_basic_$sym.txt"
            Downloads.download("$url/$file", "$path/$file")
        end
    end
    return nothing
end

"Extract data from Vreman."
function vremanstatistics()
    ufile = joinpath(@__DIR__, "Chan180_FD2_all/Chan180_FD2_basic_u.txt")
    vfile = joinpath(@__DIR__, "Chan180_FD2_all/Chan180_FD2_basic_v.txt")
    wfile = joinpath(@__DIR__, "Chan180_FD2_all/Chan180_FD2_basic_w.txt")
    udata = readdlm(ufile, comments = true, comment_char = '%')
    vdata = readdlm(vfile, comments = true, comment_char = '%')
    wdata = readdlm(wfile, comments = true, comment_char = '%')
    statistics = (;
        ycenter = udata[:, 1],
        yedge = vdata[:, 1],
        ubar = udata[:, 2],
        vbar = vdata[:, 2],
        wbar = wdata[:, 2],
        urms = udata[:, 3],
        vrms = vdata[:, 3],
        wrms = wdata[:, 3],
        up_up_up = udata[:, 4],
        up_up_up_up = udata[:, 5],
        up_up_vp = udata[:, 6],
        up_wp = udata[:, 7],
        visc = zero(udata[:, 1]), # DNS: No eddy viscosity in Vreman's data
        label = "Vreman and Kuerten (2014)",
    )
    return statistics
end

"Make wall profile plot."
function plot_wall_profile(stats, problem, doscatter)
    (; u_tau, H) = problem
    for (key, title) in [
            :ubar => L"\langle u \rangle",
            :vbar => L"\langle v \rangle",
            :wbar => L"\langle w \rangle",
            :urms => L"\sqrt{\langle u' u' \rangle}",
            :vrms => L"\sqrt{\langle v' v' \rangle}",
            :wrms => L"\sqrt{\langle w' w' \rangle}",
            :up_up_up => L"\langle u' u' u' \rangle",
            :up_up_up_up => L"\langle u' u' u' u' \rangle",
            :up_up_vp => L"\langle u' u' v' \rangle",
            :up_wp => L"\langle u' w' \rangle",
            :visc => L"\langle \nu^\Delta \rangle / \nu",
            ## :visc => "Eddy-viscosity",
        ]
        fig = Figure()
        ax = Axis(
            fig[1, 1];
            xlabelsize = 24,
            ylabelsize = 24,
            xlabel = L"y^{+}",
            ylabel = title,
            xscale = log10,
        )
        markers = [:circle, :rect, :diamond, :cross, :xcross, :utriangle, :dtriangle]
        normalizations = (;
                      ## visc = u_tau * H,
                      visc = problem.viscosity,
                     )
        for (i, s) in enumerate(stats)
            if key == :ubar && i == 1
                ## Add reference lines for mean velocity profile
                yvisc = filter(<=(18), s.ycenter)
                ylog = filter(y -> 1 <= y <= 180, s.ycenter)
                ulog = @. log(ylog) / 0.41 + 5.7
                lines!(yvisc, yvisc; color = :black, linestyle = :dash, label = "Linear profile")
                lines!(ylog, ulog; color = :black, linestyle = :dot, label = "Logarithmic profile")
            end
            yuse = key == :vrms || key == :vbar ? s.yedge : s.ycenter
            normal = haskey(normalizations, key) ? normalizations[key] : 1.0
            doscatter && scatter!(yuse, s[key] / normal; marker = markers[i], s.label)
            doscatter || lines!(yuse, s[key] / normal; s.label)
        end
        Legend(fig[1, 2], ax)
        ## path = "~/Projects/Thesis/Figures/Software" |> expanduser
        plotfile = joinpath(getoutdir(problem), "wallplot-$(key).pdf")
        println("Saving plot to: $plotfile")
        save(plotfile, fig; backend = CairoMakie, size = (700, 400))
    end
    return nothing
end

"Make wall profile RMS comparison plot."
function plot_wall_profile_rms_comparison(stats, doscatter)
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = L"y^{+}",
        ylabel = "RMS",
        xlabelsize = 24,
        title = "Comparison of velocity fluctuations",
        xscale = log10,
    )
    for (i, s) in enumerate(stats)
        if doscatter
            markers = [:circle, :rect, :diamond, :cross, :xcross, :utriangle, :dtriangle]
            scatter!(s.ycenter, s.urms; color = Cycled(i), marker = markers[1], label = L"%$(s.label), $\sqrt{\langle u' u' \rangle}$")
            scatter!(s.yedge, s.vrms; color = Cycled(i), marker = markers[2], label = L"%$(s.label), $\sqrt{\langle v' v' \rangle}$")
            scatter!(s.ycenter, s.wrms; color = Cycled(i), marker = markers[3], label = L"%$(s.label), $\sqrt{\langle w' w' \rangle}$")
        else
            linestyles = [:solid, :dash, :dot, :dashdot]
            lines!(s.ycenter, s.urms; color = Cycled(i), linestyle = linestyles[1], label = L"%$(s.label), $\sqrt{\langle u' u' \rangle}$")
            lines!(s.yedge, s.vrms; color = Cycled(i), linestyle = linestyles[2], label = L"%$(s.label), $\sqrt{\langle v' v' \rangle}$")
            lines!(s.ycenter, s.wrms; color = Cycled(i), linestyle = linestyles[3], label = L"%$(s.label), $\sqrt{\langle w' w' \rangle}$")
        end
    end
    Legend(fig[1, 2], ax)
    plotfile = joinpath(getoutdir(problem), "wallplot-comparision-rms.pdf")
    println("Saving plot to: $plotfile")
    save(plotfile, fig; backend = CairoMakie, size = (700, 400))
    return fig
end

function getoutdir(problem)
    (; nx, ny, nz, stretch, twarmup, taverage) = problem
    name = "ChannelVreman-nx=$nx-ny=$ny-nz=$nz-stretch=$stretch-twarmup=$twarmup-taverage=$taverage"
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
    @info @sprintf("Max grid spacing in wall units: Δx+ = %.4g, Δy+ = %.4g, Δz+ = %.4g\n", Δxp_max, Δyp_max, Δzp_max)
    @info @sprintf("Min grid spacing in wall units: Δx+ = %.4g, Δy+ = %.4g, Δz+ = %.4g\n", Δxp_min, Δyp_min, Δzp_min)
    @info "Grid size: nx = $nx, ny = $ny, nz = $nz"
    flush(stderr) # Prevent logging delay on Snellius
    return nothing
end

# Main script

problem = getproblem()
setup = getsetup(problem)
show_problem(setup, problem)

if problem.dosimulation
    (; psolver, ustart) = getheavystuff(setup, problem)
end

problem.dosimulation && solve(
    setup, psolver, ustart, force_nomo!,
    (; problem.viscosity, problem.forcing), problem,
    "statseries_nomo.jld2", "No-model"
)
problem.dosimulation && problem.doles && solve(
    setup, psolver, ustart, force_eddy!,
    (; problem.viscosity, problem.forcing, eddyviscosity = NS.Smagorinsky(problem.C_smag)), problem,
    "statseries_smag.jld2", "Smagorinsky"
)
problem.dosimulation && problem.doles && solve(
    setup, psolver, ustart, force_eddy!, (; problem.viscosity, problem.forcing, eddyviscosity = NS.WALE(problem.C_wale)), problem,
    "statseries_wale.jld2", "WALE",
)
problem.dosimulation && problem.doles &&
    solve(
    setup, psolver, ustart, force_eddy!,
    (; problem.viscosity, problem.forcing, eddyviscosity = NS.QR(problem.C_qr)), problem,
    "statseries_qr.jld2", "QR",
)
problem.dosimulation && problem.doles && solve(
    setup, psolver, ustart, force_eddy!,
    (; problem.viscosity, problem.forcing, eddyviscosity = NS.Vreman(problem.C_vreman)), problem,
    "statseries_vreman.jld2", "Vreman",
)

downloaddata()
statistics_ref = vremanstatistics()

if problem.doles
    statseries_nomo = load_object(joinpath(getoutdir(problem), "statseries_nomo.jld2"))
    statseries_smag = load_object(joinpath(getoutdir(problem), "statseries_smag.jld2"))
    statseries_wale = load_object(joinpath(getoutdir(problem), "statseries_wale.jld2"))
    statseries_qr = load_object(joinpath(getoutdir(problem), "statseries_qr.jld2"))
    statseries_vreman = load_object(joinpath(getoutdir(problem), "statseries_vreman.jld2"))

    statistics_nomo = process_statseries(statseries_nomo, setup, problem, "No-model")
    statistics_smag = process_statseries(statseries_smag, setup, problem, "Smagorinsky")
    statistics_wale = process_statseries(statseries_wale, setup, problem, "WALE")
    statistics_qr = process_statseries(statseries_qr, setup, problem, "QR")
    statistics_vreman = process_statseries(statseries_vreman, setup, problem, "Vreman")
    stats = [statistics_ref, statistics_nomo, statistics_smag, statistics_wale, statistics_qr, statistics_vreman]
else
    statseries_nomo = load_object(joinpath(getoutdir(problem), "statseries_nomo.jld2"))
    statistics_nomo = process_statseries(statseries_nomo, setup, problem, "IncompressibleNavierStokes.jl")
    stats = [statistics_ref, statistics_nomo]
end

doscatter = true
plot_wall_profile(stats, problem, doscatter)
plot_wall_profile_rms_comparison(stats, doscatter)

@info "Done."
flush(stdout) # Prevent logging delay on Snellius
flush(stderr) # Prevent logging delay on Snellius
