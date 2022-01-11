"""
    initialize!(processor, stepper)

Initialize processor.
"""
initialize!(logger::Logger, stepper) = logger

function initialize!(plotter::RealTimePlotter, stepper)
    (; V, p, t, setup) = stepper
    (; bc) = setup
    (; xlims, ylims, x, y, xp, yp) = setup.grid
    (; fieldname) = plotter

    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]

    vel = nothing
    ω = nothing
    ψ = nothing
    pres = nothing

    refsize = 2000

    fig = Figure(resolution = (refsize * Lx / (Lx + Ly), refsize * Ly / (Lx + Ly) + 100))
    if fieldname == :velocity
        vels = get_velocity(V, t, setup)
        qp = .√sum(vels .^ 2)

        vel = Observable(qp)
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
        ω = Observable(get_vorticity(V, t, setup))
        # ax, hm = contour(fig[1, 1], xω, yω, ω; levels = -10:2:10)
        # ax, hm = contourf(fig[1, 1], xω, yω, ω; levels = -10:2:10, extendlow = :auto, extendhigh = :auto)
        # ax, hm = heatmap(fig[1, 1], xω, yω, ω; colorrange = (-20, 20))#, colormap = :vangogh)
        ax, hm = heatmap(fig[1, 1], xω, yω, ω; colorrange = (-10, 10))#, colormap = :vangogh)
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
        ψ = Observable(get_streamfunction(V, t, setup))
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
    (; dir, filename) = writer
    ispath(dir) || mkpath(dir);
    pvd = paraview_collection(joinpath(dir, filename))
    @pack! writer = pvd
    writer
end

function initialize!(tracer::QuantityTracer, stepper)
    tracer.t = zeros(0)
    tracer.maxdiv = zeros(0)
    tracer.umom = zeros(0)
    tracer.vmom = zeros(0)
    tracer.wmom = zeros(0)
    tracer.k = zeros(0)
    tracer
end
