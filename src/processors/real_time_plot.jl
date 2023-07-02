"""
    real_time_plot(
        o,
        setup,
        fieldname = :vorticity,
        type = nothing,
        sleeptime = 0.001,
        alpha = 0.05,
    )

Plot the solution every time the state `o` is updated.

The `sleeptime` is slept at every update, to give Makie time to update the
plot. Set this to `nothing` to skip sleeping.

Available fieldnames are:

- `:velocity`,
- `:vorticity`,
- `:streamfunction`,
- `:pressure`.

Available plot `type`s for 2D are:

- `heatmap` (default),
- `contour`,
- `contourf`.

Available plot `type`s for 3D are:

- `contour` (default).

The `alpha` value gets passed to `contour` in 3D.
"""
real_time_plot(
    o,
    setup;
    kwargs...,
) = real_time_plot(setup.grid.dimension, o, setup; kwargs...)


function real_time_plot(
    ::Dimension{2},
    o,
    setup;
    fieldname = :vorticity,
    type = heatmap,
    sleeptime = 0.001,
)
    (; boundary_conditions, grid) = setup
    (; xlims, ylims, x, y, xp, yp) = grid

    if fieldname == :velocity
        xf, yf = xp, yp
    elseif fieldname == :vorticity
        if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
            xf = x
            yf = y
        else
            xf = x[2:(end-1)]
            yf = y[2:(end-1)]
        end
    elseif fieldname == :streamfunction
        if boundary_conditions.u.x[1] == :periodic
            xf = x
        else
            xf = x[2:(end-1)]
        end
        if boundary_conditions.v.y[1] == :periodic
            yf = y
        else
            yf = y[2:(end-1)]
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
            up, vp = get_velocity(setup, V, t)
            map((u, v) -> √sum(u^2 + v^2), up, vp)
        elseif fieldname == :vorticity
            get_vorticity(setup, V, t)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, V, t)
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

function real_time_plot(
    ::Dimension{3},
    o,
    setup;
    fieldname = :vorticity,
    sleeptime = 0.001,
    alpha = 0.05,
)
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
            xf = x[2:(end-1)]
            yf = y[2:(end-1)]
            zf = z[2:(end-1)]
        end
    elseif fieldname == :streamfunction
        if boundary_conditions.u.x[1] == :periodic
            xf = x
        else
            xf = x[2:(end-1)]
        end
        if boundary_conditions.v.y[1] == :periodic
            yf = y
        else
            yf = y[2:(end-1)]
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
            up, vp, wp = get_velocity(setup, V, t)
            map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
        elseif fieldname == :vorticity
            get_vorticity(setup, V, t)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, V, t)
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

"""
    energy_history_plot(o, setup)

Create energy history plot, with a history point added every time `o` is updated.
"""

function energy_history_plot(o::StateObserver, setup)
    (; Ωp) = setup.grid
    _points = Point2f[]
    points = @lift begin
        V, p, t = $(o.state)
        vels = get_velocity(setup, V, t)
        vels = reshape.(vels, :)
        E = sum(vel -> sum(@. Ωp * vel^2), vels)
        push!(_points, Point2f(t, E))
    end
    lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))
end

"""
    energy_spectrum_plot(o, setup, K)

Create energy spectrum plot, redrawn every time `o` is updated.
"""
function energy_spectrum_plot end

energy_spectrum_plot(o, setup, K) = energy_spectrum_plot(setup.grid.dimension, o, setup, K)

function energy_spectrum_plot(
    ::Dimension{2},
    o,
    setup,
    K,
)
    k = 1:(K-1)
    kk = reshape([sqrt(kx^2 + ky^2) for kx ∈ k, ky ∈ k], :)
    ehat = @lift begin
        V, p, t = $(o.state)
        up, vp = get_velocity(setup, V, t)
        e = up .^ 2 .+ vp .^ 2
        reshape(abs.(fft(e)[k.+1, k.+1]), :)
    end
    espec = Figure()
    ax =
        Axis(espec[1, 1]; xlabel = "k", ylabel = "e(k)", xscale = log10, yscale = log10)
    ## ylims!(ax, (1e-20, 1))
    scatter!(ax, kk, ehat; label = "Kinetic energy")
    krange = LinRange(extrema(kk)..., 100)
    lines!(ax, krange, 1e7 * krange .^ (-3); label = "k⁻³", color = :red)
    axislegend(ax)
    espec
end

function energy_spectrum_plot(
    ::Dimension{3},
    o,
    setup,
    K,
)
    k = 1:(K-1)
    kk = reshape([sqrt(kx^2 + ky^2 + kz^2) for kx ∈ k, ky ∈ k, kz ∈ k], :)
    ehat = @lift begin
        V, p, t = $(o.state)
        up, vp, wp = get_velocity(setup, V, t)
        e = @. up^2 + vp^2 + wp^2
        reshape(abs.(fft(e)[k.+1, k.+1, k.+1]), :)
    end
    espec = Figure()
    ax =
        Axis(espec[1, 1]; xlabel = "k", ylabel = "e(k)", xscale = log10, yscale = log10)
    ## ylims!(ax, (1e-20, 1))
    scatter!(ax, kk, ehat; label = "Kinetic energy")
    krange = LinRange(extrema(kk)..., 100)
    lines!(ax, krange, 1e6 * krange .^ (-5 / 3); label = "\$k^{-5/3}\$", color = :red)
    axislegend(ax)
    espec
end