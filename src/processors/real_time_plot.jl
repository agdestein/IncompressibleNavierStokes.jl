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
- `image`,
- `contour`,
- `contourf`.

Available plot `type`s for 3D are:

- `contour` (default).

The `alpha` value gets passed to `contour` in 3D.
"""
field_plotter(setup; nupdate = 1, kwargs...) =
    processor(state -> field_plot(setup.grid.dimension, setup, state; kwargs...); nupdate)

function field_plot(
    ::Dimension{2},
    setup,
    state;
    fieldname = :vorticity,
    type = heatmap,
    sleeptime = 0.001,
    equal_axis = true,
    displayfig = true,
)
    (; boundary_conditions, grid) = setup
    (; dimension, xlims, x, xp, Ip) = grid
    D = dimension()

    if fieldname == :velocity
        xf = xp
    elseif fieldname == :vorticity
        xf = ntuple(α -> Array(xp[α][Ip.indices[α]]), D)
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
        xf = xp
    else
        error("Unknown fieldname")
    end

    field = @lift begin
        isnothing(sleeptime) || sleep(sleeptime)
        (; u, p, t) = $state
        f = if fieldname == :velocity
            up, vp = get_velocity(setup, u, t)
            map((u, v) -> √sum(u^2 + v^2), up, vp)
        elseif fieldname == :vorticity
            interpolate_ω_p(setup, vorticity(u, setup))[Ip]
        elseif fieldname == :streamfunction
            get_streamfunction(setup, u, t)
        elseif fieldname == :pressure
            error("Not implemented")
            reshape(p, length(xp), length(yp))
        end
        Array(f)
    end

    lims = @lift begin
        f = $field
        if type ∈ (heatmap, image)
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

    if type ∈ (heatmap, image)
        ax, hm = type(fig[1, 1], xf..., field; colormap = :viridis, colorrange = lims)
    elseif type ∈ (contour, contourf)
        ax, hm = type(
            fig[1, 1],
            xf...,
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
    equal_axis && (ax.aspect = DataAspect())
    ax.xlabel = "x"
    ax.ylabel = "y"
    limits!(ax, xlims[1]..., xlims[2]...)
    Colorbar(fig[1, 2], hm)

    displayfig && display(fig)

    fig
end

function field_plot(
    ::Dimension{3},
    setup,
    state;
    fieldname = :vorticity,
    sleeptime = 0.001,
    alpha = 0.05,
    equal_axis = true,
    levels = 3,
    displayfig = true,
)
    (; boundary_conditions, grid) = setup
    (; xlims, x, xp) = grid

    if fieldname == :velocity
        xf = xp
    elseif fieldname == :vorticity
            xf = xp
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
        (; u, p, t) = $state
        f = if fieldname == :velocity
            up = interpolate_u_p(setup, u)
            map((u, v, w) -> √sum(u^2 + v^2 + w^2), up...)
        elseif fieldname == :vorticity
            ωp = interpolate_ω_p(setup, vorticity(u, setup))
            map((u, v, w) -> √sum(u^2 + v^2 + w^2), ωp...)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, u, t)
        elseif fieldname == :pressure
            reshape(copy(p), length(xp), length(yp), length(zp))
        end
        Array(f)
    end

    lims = @lift get_lims($field)

    isnothing(levels) && (levels = @lift(LinRange($(lims)..., 10)))

    aspect = equal_axis ? (; aspect = :data) : (;)
    fig = Figure()
    ax = Axis3(fig[1, 1]; title = titlecase(string(fieldname)), aspect...)
    hm = contour!(
        ax,
        xf...,
        field;
        # levels,
        # colorrange = lims,
        shading = false,
        alpha,
        highclip = :red,
        lowclip = :red,
    )

    Colorbar(fig[1, 2], hm)

    displayfig && display(fig)

    fig
end

"""
    energy_history_plotter(setup)

Create energy history plot, with a history point added every time `step_observer` is updated.
"""
energy_history_plotter(setup; nupdate = 1, kwargs...) =
    processor(state -> energy_history_plot(setup, state; kwargs...); nupdate)

function energy_history_plot(setup, state; displayfig = true)
    _points = Point2f[]
    points = @lift begin
        (; u, p, t) = $state
        E = kinetic_energy(setup, u)
        push!(_points, Point2f(t, E))
    end
    fig = lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))
    displayfig && display(fig)
    on(_ -> autolimits!(fig.axis), points)
    fig
end

"""
    energy_spectrum_plotter(setup; nupdate = 1)

Create energy spectrum plot, redrawn every time `step_observer` is updated.
"""
energy_spectrum_plotter(setup; nupdate = 1, kwargs...) = processor(
    state -> energy_spectrum_plot(setup, state; kwargs...);
    nupdate,
)

function energy_spectrum_plot(setup, state; displayfig = true)
    (; dimension, xp, Ip) = setup.grid
    T = eltype(xp[1])
    D = dimension()
    K = size(Ip) .÷ 2
    kx = ntuple(α -> 1:K[α]-1, D)
    k = KernelAbstractions.zeros(get_backend(xp[1]), T, length.(kx)...)
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = Array(reshape(k, :))
    ehat = @lift begin
        (; u, p, t) = $state
        up = interpolate_u_p(setup, u)
        e = sum(up -> up[Ip] .^ 2, up)
        Array(reshape(abs.(fft(e)[ntuple(α -> kx[α].+1, D)...]), :))
    end
    espec = Figure()
    ax = Axis(espec[1, 1]; xlabel = "k", ylabel = "e(k)", xscale = log10, yscale = log10)
    ## ylims!(ax, (1e-20, 1))
    scatter!(ax, k, ehat; label = "Kinetic energy")
    krange = LinRange(extrema(k)..., 100)
    D == 2 && lines!(ax, krange, 1e7 * krange .^ (-3); label = "k⁻³", color = :red)
    D == 3 && lines!(ax, krange, 1e6 * krange .^ (-5 / 3); label = "\$k^{-5/3}\$", color = :red)
    axislegend(ax)
    displayfig && display(espec)
    on(ehat) do _
        autolimits!(ax)
    end
    espec
end

# # Make sure the figure is fully rendered before allowing code to continue
# if displayfig
#     render = display(espec)
#     done_rendering = Ref(false)
#     on(render.render_tic) do _
#         done_rendering[] = true
#     end
#     on(state) do s
#         # State is updated, block code execution until GLMakie has rendered
#         # figure update
#         done_rendering[] = false
#         while !done_rendering[]
#             sleep(checktime)
#         end
#     end
# end
