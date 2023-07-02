"""
    field_plotter(
        setup;
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
field_plotter(setup; nupdate = 1, kwargs...) = processor(
    state -> field_plot(setup.grid.dimension, setup, state; kwargs...);
    nupdate,
)

function field_plot(
    ::Dimension{2},
    setup,
    state;
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
        (; V, p, t) = $state
        f = if fieldname == :velocity
            up, vp = get_velocity(setup, V, t)
            map((u, v) -> √sum(u^2 + v^2), up, vp)
        elseif fieldname == :vorticity
            get_vorticity(setup, V, t)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, V, t)
        elseif fieldname == :pressure
            error("Not implemented")
            reshape(p, length(xp), length(yp))
        end
        Array(f)
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

    display(fig)

    fig
end

function field_plot(
    ::Dimension{3},
    setup,
    state;
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
        (; V, p, t) = $state
        f = if fieldname == :velocity
            up, vp, wp = get_velocity(setup, V, t)
            map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
        elseif fieldname == :vorticity
            get_vorticity(setup, V, t)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, V, t)
        elseif fieldname == :pressure
            reshape(copy(p), length(xp), length(yp), length(zp))
        end
        Array(f)
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

    display(fig)

    fig
end

"""
    energy_history_plotter(setup)

Create energy history plot, with a history point added every time `step_observer` is updated.
"""
energy_history_plotter(setup; nupdate = 1, kwargs...) = processor(
    state -> energy_history_plot(setup, state; kwargs...);
    nupdate,
)

function energy_history_plot(setup, state)
    (; Ωp) = setup.grid
    _points = Point2f[]
    points = @lift begin
        (; V, p, t) = $state
        vels = get_velocity(setup, V, t)
        vels = reshape.(vels, :)
        E = sum(vel -> sum(@. Ωp * vel^2), vels)
        push!(_points, Point2f(t, E))
    end
    fig = lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))
    display(fig)
    fig
end

"""
    energy_spectrum_plotter(setup; nupdate = 1)

Create energy spectrum plot, redrawn every time `step_observer` is updated.
"""
energy_spectrum_plotter(setup; nupdate = 1, kwargs...) = processor(
    state -> energy_spectrum_plot(setup.grid.dimension, setup, state; kwargs...);
    nupdate,
)

function energy_spectrum_plot(
    ::Dimension{2},
    setup,
    state,
)
    (; xpp) = setup.grid
    Kx, Ky = size(xpp) .÷ 2
    kx = 1:(Kx-1)
    ky = 1:(Ky-1)
    kk = reshape([sqrt(kx^2 + ky^2) for kx ∈ kx, ky ∈ ky], :)
    ehat = @lift begin
        (; V, p, t) = $state
        up, vp = get_velocity(setup, V, t)
        e = up .^ 2 .+ vp .^ 2
        reshape(abs.(fft(e)[kx.+1, ky.+1]), :)
    end
    espec = Figure()
    ax =
        Axis(espec[1, 1]; xlabel = "k", ylabel = "e(k)", xscale = log10, yscale = log10)
    ## ylims!(ax, (1e-20, 1))
    scatter!(ax, kk, ehat; label = "Kinetic energy")
    krange = LinRange(extrema(kk)..., 100)
    lines!(ax, krange, 1e7 * krange .^ (-3); label = "k⁻³", color = :red)
    axislegend(ax)
    display(espec)
    espec
end

function energy_spectrum_plot(
    ::Dimension{3},
    state,
    setup,
    K,
)
    (; xpp) = setup.grid
    Kx, Ky, Kz = size(xpp) .÷ 2
    kx = 1:(Kx-1)
    ky = 1:(Ky-1)
    kz = 1:(Ky-1)
    kk = reshape([sqrt(kx^2 + ky^2 + kz^2) for kx ∈ kx, ky ∈ ky, kz ∈ kz], :)
    ehat = @lift begin
        V, p, t = $(state.state)
        up, vp, wp = get_velocity(setup, V, t)
        e = @. up^2 + vp^2 + wp^2
        reshape(abs.(fft(e)[kx.+1, ky.+1, kz.+1]), :)
    end
    espec = Figure()
    ax =
        Axis(espec[1, 1]; xlabel = "k", ylabel = "e(k)", xscale = log10, yscale = log10)
    ## ylims!(ax, (1e-20, 1))
    scatter!(ax, kk, ehat; label = "Kinetic energy")
    krange = LinRange(extrema(kk)..., 100)
    lines!(ax, krange, 1e6 * krange .^ (-5 / 3); label = "\$k^{-5/3}\$", color = :red)
    axislegend(ax)
    display(espec)
    espec
end
