"""
    initialize!(processor, stepper)

Initialize processor.
"""
initialize!(logger::Logger, stepper) = logger

# 2D real time plot
function initialize!(plotter::RealTimePlotter, stepper::AbstractTimeStepper{T,2}) where {T}
    (; V, p, t, setup) = stepper
    (; grid) = setup
    (; xlims, ylims, x, y, xp, yp) = grid
    (; fieldname, type) = plotter

    if fieldname == :velocity
        up, vp = get_velocity(V, t, setup)
        f = map((u, v) -> √sum(u^2 + v^2), up, vp)
        xf, yf = xp, yp
    elseif fieldname == :vorticity
        xf = x
        yf = y
        f = get_vorticity(V, t, setup)
    elseif fieldname == :streamfunction
        xf = x
        yf = y
        f = get_streamfunction(V, t, setup)
    elseif fieldname == :pressure
        error("Not implemented")
        xf, yf = xp, yp
        f = reshape(copy(p), length(xp), length(yp))
    else
        error("Unknown fieldname")
    end

    field = Observable(f)

    fig = Figure()

    if type == heatmap
        lims = Observable(get_lims(f))
        ax, hm = heatmap(fig[1, 1], xf, yf, field; colorrange = lims)
    elseif type ∈ (contour, contourf)
        lims = Observable(LinRange(get_lims(f)..., 10))
        ax, hm = type(
            fig[1, 1],
            xf,
            yf,
            field;
            extendlow = :auto,
            extendhigh = :auto,
            levels = lims,
        )
    else
        error("Unknown plot type")
    end

    ax.title = titlecase(string(fieldname))
    # ax.aspect = DataAspect()
    ax.xlabel = "x"
    ax.ylabel = "y"
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    Colorbar(fig[1, 2], hm)
    display(fig)

    @pack! plotter = field, lims

    plotter
end

# 3D real time plot
function initialize!(plotter::RealTimePlotter, stepper::AbstractTimeStepper{T,3}) where {T}
    (; V, p, t, setup) = stepper
    (; grid) = setup
    (; x, y, z, xp, yp, zp) = grid
    (; fieldname, type) = plotter

    if fieldname == :velocity
        up, vp, wp = get_velocity(V, t, setup)
        f = map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
        xf, yf, zf = xp, yp, zp
    elseif fieldname == :vorticity
        xf = x
        yf = y
        zf = y
        f = get_vorticity(V, t, setup)
    elseif fieldname == :streamfunction
        xf = x
        yf = y
        f = get_streamfunction(V, t, setup)
    elseif fieldname == :pressure
        xf, yf, zf = xp, yp, zp
        f = reshape(copy(p), length(xp), length(yp), length(zp))
    else
        error("Unknown fieldname")
    end

    field = Observable(f)

    fig = Figure()

    if type ∈ (contour, contourf)
        type == contour && (type! = contour!)
        type == contourf && (type! = contourf!)
        lims = Observable(LinRange(get_lims(f, 3)..., 10))
        ax = Axis3(fig[1, 1]; title = titlecase(string(fieldname)), aspect = :data)
        hm = type!(ax, xf, yf, zf, field; levels = lims, shading = false, alpha = 0.05)
    else
        error("Unknown plot type")
    end
    # Colorbar(fig[1, 2], hm; ticks = lims)
    display(fig)

    @pack! plotter = field, lims

    plotter
end

function initialize!(writer::VTKWriter, stepper)
    (; dir, filename) = writer
    ispath(dir) || mkpath(dir)
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
