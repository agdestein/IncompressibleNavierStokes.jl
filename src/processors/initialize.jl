"""
    initialize!(processor, stepper)

Initialize processor.
"""
initialize!(logger::Logger, stepper) = logger

# 2D real time plot
function initialize!(plotter::RealTimePlotter, stepper::TimeStepper{M,T,2}) where {M,T}
    (; V, p, t, setup) = stepper
    (; bc) = setup
    (; xlims, ylims, x, y, xp, yp) = setup.grid
    (; fieldname) = plotter

    vel = nothing
    ω = nothing
    ψ = nothing

    fig = Figure()
    if fieldname == :velocity
        up, vp = get_velocity(V, t, setup)
        qp = map((u, v) -> √sum(u ^ 2 + v ^ 2), up, vp)

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

# 3D real time plot
function initialize!(plotter::RealTimePlotter, stepper::TimeStepper{M,T,3}) where {M,T}
    (; V, t, setup) = stepper
    (; xlims, ylims, zlims, xp, yp, zp) = setup.grid
    (; fieldname) = plotter


    fig = Figure()
    if fieldname == :velocity
        up, vp, wp = get_velocity(V, t, setup)
        qp = map((u, v, w) -> √sum(u ^ 2 + v ^ 2 + w ^ 2), up, vp, wp)
        vel = Observable(qp)
        field = vel
        ax = Axis3(fig[1, 1]; title = "Velocity (magnitude)", aspect = :data)
        hm = contour!(ax, xp, yp, zp, vel; shading = false)
    else
        error("Unknown fieldname")
    end
    # Colorbar(fig[1, 2], hm)
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
