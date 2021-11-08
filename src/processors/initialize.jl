"""
    initialize!(processor, stepper)

Initialize processor.
"""
function initialize!(logger::Logger, stepper)
    @unpack V, p, t, setup, cache, momentum_cache = stepper
    @unpack t_start, t_end, Δt = setup.time
    @unpack F = cache

    # Estimate number of time steps that will be taken
    nt = ceil(Int, (t_end - t_start) / Δt)

    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)
    maxres = maximum(abs.(F))

    println("n = $(stepper.n), t = $t, maxres = $maxres")
    # println("t = $t")

    logger
end

function initialize!(plotter::RealTimePlotter, stepper)
    @unpack V, p, t, setup = stepper
    @unpack bc = setup
    @unpack xlims, ylims, Nx, Ny, x, y, xp, yp = setup.grid
    @unpack fieldname = plotter

    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]

    vel = nothing
    ω = nothing
    ψ = nothing
    pres = nothing

    refsize = 2000

    fig = Figure(resolution = (refsize * Lx / (Lx + Ly), refsize * Ly / (Lx + Ly) + 100))
    if fieldname == :velocity
        up, vp, qp = get_velocity(V, t, setup)
        vel = Node(qp)
        ax, hm = contourf(fig[1, 1], xp, yp, vel)
        field = vel
    elseif fieldname == :vorticity
        if bc.u.x[1] == :periodic
            xω = x
        else
            xω = x[2:end-1]
        end
        if bc.v.y[1] == :periodic
            yω = y
        else
            yω = y[2:end-1]
        end
        ω = Node(get_vorticity(V, t, setup))
        # ax, hm = contour(fig[1, 1], xω, yω, ω; levels = -10:2:10)
        # ax, hm = contourf(fig[1, 1], xω, yω, ω; levels = -10:2:10, extendlow = :auto, extendhigh = :auto)
        ax, hm = heatmap(fig[1, 1], xω, yω, ω; colorrange = (-20, 20))#, colormap = :vangogh)
        field = ω
    elseif fieldname == :streamfunction
        if bc.u.x[1] == :periodic
            xψ = x
        else
            xψ = x[2:end-1]
        end
        if bc.v.y[1] == :periodic
            yψ = y
        else
            yψ = y[2:end-1]
        end
        ψ = Node(get_streamfunction(V, t, setup))
        ax, hm = contourf(fig[1, 1], xψ, yψ, ψ)
        field = ψ
    elseif fieldname == :pressure
        error("Not implemented")
        field = p
    else
        error("Unknown fieldname")
    end
    ax.title = titlecase(string(fieldname))
    ax.aspect = DataAspect()
    ax.xlabel = "x"
    ax.ylabel = "y"
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    Colorbar(fig[1, 2], hm)
    display(fig)

    @pack! plotter = field

    plotter
end

function initialize!(writer::VTKWriter, stepper)
    @unpack dir, filename = writer
    isdir(dir) || mkdir(dir);
    pvd = paraview_collection(joinpath(dir, filename))
    @pack! writer = pvd

    writer
end
