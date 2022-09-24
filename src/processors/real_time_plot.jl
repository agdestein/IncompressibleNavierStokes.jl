function real_time_plot end

"""
    real_time_plot(
        state_observer,
        setup::Setup{T,2};
        fieldname = :vorticity,
        type = heatmap,
        sleeptime = 0.001,
    )

Plot the solution every time the state is updated (2D version).

The `sleeptime` is slept at every update, to give Makie time to update the
plot. Set this to `nothing` to skip sleeping.

Available fieldnames are:

- `:velocity`,
- `:vorticity`,
- `:streamfunction`,
- `:pressure`.

Available plot `type`s are:

- `heatmap`,
- `contour`,
- `contourf`.
"""
function real_time_plot(
    o::StateObserver,
    setup::Setup{T,2};
    fieldname = :vorticity,
    type = heatmap,
    sleeptime = 0.001,
) where {T}
    (; boundary_conditions, grid) = setup
    (; xlims, ylims, x, y, xp, yp) = grid

    if fieldname == :velocity
        xf, yf = xp, yp
    elseif fieldname == :vorticity
        if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
            xf = x
            yf = y
        else
            xf = x[2:(end - 1)]
            yf = y[2:(end - 1)]
        end
    elseif fieldname == :streamfunction
        if boundary_conditions.u.x[1] == :periodic
            xf = x
        else
            xf = x[2:(end - 1)]
        end
        if boundary_conditions.v.y[1] == :periodic
            yf = y
        else
            yf = y[2:(end - 1)]
        end
    elseif fieldname == :pressure
        error("Not implemented")
        xf, yf = xp, yp
    else
        error("Unknown fieldname")
    end

    field = @lift begin
        isnothing(sleeptime) || sleep(sleeptime)
        (V, p, t) = $(o.state)
        if fieldname == :velocity
            up, vp = get_velocity(V, t, setup)
            map((u, v) -> √sum(u^2 + v^2), up, vp)
        elseif fieldname == :vorticity
            get_vorticity(V, t, setup)
        elseif fieldname == :streamfunction
            get_streamfunction(V, t, setup)
        elseif fieldname == :pressure
            error("Not implemented")
            reshape(copy(p), length(xp), length(yp))
        end
    end

    lims = @lift begin
        f = $field
        if type == heatmap
            lims = get_lims(f)
        elseif type ∈ (contour, contourf)
            if ≈(extrema(f)...; rtol = 1e-10)
                μ = mean(f)
                a = μ - 1
                b = μ + 1
                f[1] += 1
                f[end] -= 1
            else
                a, b = get_lims(f)
            end
            lims = (a, b)
        end
        lims
    end

    fig = Figure()

    if type == heatmap
        ax, hm = heatmap(fig[1, 1], xf, yf, field; colorrange = lims)
    elseif type ∈ (contour, contourf)
        ax, hm = type(
            fig[1, 1],
            xf,
            yf,
            field;
            extendlow = :auto,
            extendhigh = :auto,
            levels = @lift(LinRange($(lims)..., 10)),
            colorrange = lims,
        )
    else
        error("Unknown plot type")
    end

    ax.title = titlecase(string(fieldname))
    ax.aspect = DataAspect()
    ax.xlabel = "x"
    ax.ylabel = "y"
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    Colorbar(fig[1, 2], hm)

    fig
end

"""
    real_time_plot(
        state_observer,
        setup::Setup{T,3};
        fieldname = :vorticity,
        sleeptime = 0.001,
        alpha = 0.05,
    )

Plot the solution every time the state is updated (3D version).

The `sleeptime` is slept at every update, to give Makie time to update the
plot. Set this to `nothing` to skip sleeping.

The plot type if `contour`.

Available fieldnames are:

- `:velocity`,
- `:vorticity`,
- `:streamfunction`,
- `:pressure`.

The `alpha` value gets passed to `contour`.
"""
function real_time_plot(
    o::StateObserver,
    setup::Setup{T,3};
    fieldname = :vorticity,
    sleeptime = 0.001,
    alpha = 0.05,
) where {T}
    (; boundary_conditions, grid) = setup
    (; xlims, ylims, x, y, z, xp, yp, zp) = grid

    if fieldname == :velocity
        xf, yf, zf = xp, yp, zp
    elseif fieldname == :vorticity
        if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
            xf = x
            yf = y
            zf = y
        else
            xf = x[2:(end - 1)]
            yf = y[2:(end - 1)]
            zf = z[2:(end - 1)]
        end
    elseif fieldname == :streamfunction
        if boundary_conditions.u.x[1] == :periodic
            xf = x
        else
            xf = x[2:(end - 1)]
        end
        if boundary_conditions.v.y[1] == :periodic
            yf = y
        else
            yf = y[2:(end - 1)]
        end
    elseif fieldname == :pressure
        xf, yf, zf = xp, yp, zp
    else
        error("Unknown fieldname")
    end

    field = @lift begin
        isnothing(sleeptime) || sleep(sleeptime)
        (V, p, t) = $(o.state)
        if fieldname == :velocity
            up, vp, wp = get_velocity(V, t, setup)
            map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
        elseif fieldname == :vorticity
            get_vorticity(V, t, setup)
        elseif fieldname == :streamfunction
            get_streamfunction(V, t, setup)
        elseif fieldname == :pressure
            reshape(copy(p), length(xp), length(yp), length(zp))
        end
    end

    lims = @lift get_lims($field)

    fig = Figure()
    ax = Axis3(fig[1, 1]; title = titlecase(string(fieldname)), aspect = :data)
    hm = contour!(
        ax,
        xf,
        yf,
        zf,
        field;
        levels = @lift(LinRange($(lims)..., 10)),
        colorrange = lims,
        shading = false,
        alpha,
    )

    Colorbar(fig[1, 2], hm)

    fig
end
