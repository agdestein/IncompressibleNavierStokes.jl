"""
# Makie extension for IncompressibleNavierStokes

This module adds methods to empty plotting functions defined in
IncompressibleNavierStokes. The methods are loaded when Makie is loaded in the
environment, through GLMakie or CairoMakie. This allows for installing
IncompressibleNavierStokes without Makie on servers to reduce precompilation
time.
"""
module IncompressibleNavierStokesMakieExt

using DocStringExtensions
using IncompressibleNavierStokes
using IncompressibleNavierStokes: Dimension, kinetic_energy!, scalewithvolume!
using Makie
using Observables

# We will extend these functions
import IncompressibleNavierStokes:
    animator, realtimeplotter, fieldplot, energy_history_plot, energy_spectrum_plot

# Inherit docstring templates
@template (MODULES, FUNCTIONS, METHODS, TYPES) = IncompressibleNavierStokes

plotgrid(x, y; kwargs...) = wireframe(
    x,
    y,
    zeros(eltype(x), length(x), length(y));
    axis = (; aspect = DataAspect(), xlabel = "x", ylabel = "y"),
    kwargs...,
)

function plotgrid(x, y, z)
    nx, ny, nz = length(x), length(y), length(z)
    T = eltype(x)

    # x = repeat(x, 1, ny, nz)
    # y = repeat(reshape(y, 1, :, 1), nx, 1, nz)
    # z = repeat(reshape(z, 1, 1, :), nx, ny, 1)
    # vol = repeat(reshape(z, 1, 1, :), nx, ny, 1)
    # volume(x, y, z, vol)
    fig = Figure()

    ax = Axis3(fig[1, 1])
    wireframe!(ax, x, y, fill(z[1], length(x), length(y)))
    wireframe!(ax, x, y, fill(z[end], length(x), length(y)))
    wireframe!(ax, x, fill(y[1], length(z)), repeat(z, 1, length(x))')
    wireframe!(ax, x, fill(y[end], length(z)), repeat(z, 1, length(x))')
    wireframe!(ax, fill(x[1], length(z)), y, repeat(z, 1, length(y)))
    wireframe!(ax, fill(x[end], length(z)), y, repeat(z, 1, length(y)))
    ax.aspect = :data

    ax = Axis(fig[1, 2]; xlabel = "x", ylabel = "y")
    wireframe!(ax, x, y, zeros(T, length(x), length(y)))
    ax.aspect = DataAspect()

    ax = Axis(fig[2, 1]; xlabel = "y", ylabel = "z")
    wireframe!(ax, y, z, zeros(T, length(y), length(z)))
    ax.aspect = DataAspect()

    ax = Axis(fig[2, 2]; xlabel = "x", ylabel = "z")
    wireframe!(ax, x, z, zeros(T, length(x), length(z)))
    ax.aspect = DataAspect()

    fig
end

"""
Animate a plot of the solution every `update` iteration.
The animation is saved to `path`, which should have one
of the following extensions:

- ".mkv"
- ".mp4"
- ".webm"
- ".gif"

The plot is determined by a `plotter` processor.
Additional `kwargs` are passed to `plot`.
"""
animator(;
    setup,
    path,
    plot = fieldplot,
    nupdate = 1,
    framerate = 24,
    visible = true,
    screen = nothing,
    kwargs...,
) =
    processor((stream, state) -> save(path, stream)) do outerstate
        ispath(dirname(path)) || mkpath(dirname(path))
        state = Observable(outerstate[])
        fig = plot(state; setup, kwargs...)
        visible && isnothing(screen) && display(fig)
        visible && !isnothing(screen) && display(screen, fig)
        stream = VideoStream(fig; framerate, visible)
        on(outerstate) do outerstate
            outerstate.n % nupdate == 0 || return
            state[] = outerstate
            recordframe!(stream)
        end
        stream
    end

"""
Processor for plotting the solution in real time.

Keyword arguments:

- `plot`: Plot function.
- `nupdate`: Show solution every `nupdate` time step.
- `displayfig`: Display the figure at the start.
- `screen`: If `nothing`, use default display.
    If `GLMakie.screen()` multiple plots can be displayed in separate
    windows like in MATLAB (see also `GLMakie.closeall()`).
- `displayupdates`: Display the figure at every update (if using CairoMakie).
- `sleeptime`: The `sleeptime` is slept at every update, to give Makie
    time to update the plot. Set this to `nothing` to skip sleeping.

Additional `kwargs` are passed to the `plot` function.
"""
realtimeplotter(;
    setup,
    plot = fieldplot,
    nupdate = 1,
    displayfig = true,
    screen = nothing,
    displayupdates = false,
    sleeptime = nothing,
    kwargs...,
) =
    processor() do outerstate
        state = Observable(outerstate[])
        fig = plot(state; setup, kwargs...)
        displayfig && isnothing(screen) && display(fig)
        displayfig && !isnothing(screen) && display(screen, fig)
        on(outerstate) do outerstate
            outerstate.n % nupdate == 0 || return
            state[] = outerstate
            displayupdates && display(fig)
            isnothing(sleeptime) || sleep(sleeptime)
        end
        fig
    end

"""
Plot `state` field in pressure points.
If `state` is `Observable`, then the plot is interactive.

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
fieldplot(state; setup, kwargs...) = fieldplot(
    setup.grid.dimension,
    state isa Observable ? state : Observable(state);
    setup,
    kwargs...,
)

function fieldplot(
    ::Dimension{2},
    state;
    setup,
    fieldname = :vorticity,
    psolver = nothing,
    type = heatmap,
    equal_axis = true,
    docolorbar = true,
    size = nothing,
    title = nothing,
    kwargs...,
)
    (; grid) = setup
    (; dimension, xlims, xp, Ip, Δ) = grid
    D = dimension()

    xf = Array.(getindex.(xp, Ip.indices))

    field = observefield(state; setup, fieldname, psolver)

    lims = lift(field) do f
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

    if type ∈ (heatmap, image)
        kwargs = (; colorrange = lims, kwargs...)
    elseif type ∈ (contour, contourf)
        kwargs = (;
            extendlow = :auto,
            extendhigh = :auto,
            levels = @lift(LinRange($(lims)..., 10)),
            # colorrange = lims,
            kwargs...,
        )
    end

    axis = (;
        xlabel = "x",
        ylabel = "y",
        title = isnothing(title) ? titlecase(string(fieldname)) : title,
        limits = (xlims[1]..., xlims[2]...),
    )
    equal_axis && (axis = (axis..., aspect = DataAspect()))

    # Image requires boundary coordinates only
    if type == image
        Δx = first.(Array.(Δ))
        @assert all(≈(Δx[1]), Δx) "Image requires rectangular pixels"
        @assert(all(α -> all(≈(Δx[α]), Δ[α]), 1:D), "Image requires uniform grid",)
        xf = map(extrema, xf)
    end

    size = isnothing(size) ? (;) : (; size)
    fig = Figure(; size...)
    ax, hm = type(fig[1, 1], xf..., field; axis, kwargs...)
    docolorbar && Colorbar(fig[1, 2], hm)

    fig
end

function fieldplot(
    ::Dimension{3},
    state;
    setup,
    psolver = nothing,
    fieldname = :eig2field,
    alpha = convert(eltype(setup.grid.x[1]), 0.1),
    # isorange = convert(eltype(setup.grid.x[1]), 0.5),
    equal_axis = true,
    levels = LinRange{eltype(setup.grid.x[1])}(-10, 5, 10),
    docolorbar = false,
    size = nothing,
    type = contour,
    kwargs...,
)
    (; grid) = setup
    (; xp, Ip) = grid

    xf = Array.(getindex.(xp, Ip.indices))
    dxf = diff.(xf)
    xf = map(xf) do xf
        dxf = diff(xf)
        if all(≈(dxf[1]), dxf)
            LinRange(xf[1], xf[end], length(xf))
        else
            xf
        end
    end

    field = observefield(state; setup, fieldname, psolver)

    # color = lift(state) do (; temp)
    #     Array(view(temp, Ip))
    # end
    # colorrange = lift(state) do (; temp)
    #     extrema(view(temp, Ip))
    # end

    # lims = @lift get_lims($field)
    lims = isnothing(levels) ? lift(get_lims, field) : extrema(levels)

    isnothing(levels) && (levels = @lift(LinRange($(lims)..., 10)))

    # aspect = equal_axis ? (; aspect = :data) : (;)
    size = isnothing(size) ? (;) : (; size)
    fig = Figure(; size...)
    # ax = Axis3(fig[1, 1]; title = titlecase(string(fieldname)), aspect...)
    if type == volume
        hm = volume(
            fig[1, 1],
            xf...,
            field;
            # colorrange = lims,
            kwargs...,
        )
    elseif type == contour
        hm = contour(
            fig[1, 1],
            # ax,
            xf...,
            field;
            levels,
            # color = xf[2]' .+ 0 .* field[],
            # colorrange,
            colorrange = lims,
            # colorrange = extrema(levels),
            alpha,
            # isorange,
            # highclip = :red,
            # lowclip = :red,
            kwargs...,
        )
    end
    docolorbar && Colorbar(fig[1, 2], hm)
    fig
end

"""
Create energy history plot.
"""
function energy_history_plot(state; setup)
    @assert state isa Observable "Energy history requires observable state."
    (; Ip) = setup.grid
    e = scalarfield(setup)
    _points = Point2f[]
    points = lift(state) do (; u, t)
        kinetic_energy!(e, u, setup)
        scalewithvolume!(e, setup)
        E = sum(e[Ip])
        push!(_points, Point2f(t, E))
    end
    fig = lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))
    on(_ -> autolimits!(fig.axis), points)
    fig
end

"""
Create energy spectrum plot.
The energy at a scalar wavenumber level ``\\kappa \\in \\mathbb{N}`` is defined by

```math
\\hat{e}(\\kappa) = \\int_{\\kappa \\leq \\| k \\|_2 < \\kappa + 1} | \\hat{e}(k) | \\mathrm{d} k,
```

as in San and Staples [San2012](@cite).

Keyword arguments:

- `sloperange = [0.6, 0.9]`: Percentage (between 0 and 1) of x-axis where the slope is plotted.
- `slopeoffset = 1.3`: How far above the energy spectrum the inertial slope is plotted.
- `kwargs...`: They are passed to [`observespectrum`](@ref).
"""
function energy_spectrum_plot(
    state;
    setup,
    sloperange = [0.6, 0.9],
    slopeoffset = 1.3,
    kwargs...,
)
    state isa Observable || (state = Observable(state))

    (; dimension, xp, Ip) = setup.grid
    T = eltype(xp[1])
    D = dimension()

    (; ehat, κ) = observespectrum(state; setup, kwargs...)

    kmax = maximum(κ)

    # Build inertial slope above energy
    krange = kmax .^ sloperange
    slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}$") : (-T(5 / 3), L"$k^{-5/3}$")
    inertia = lift(ehat) do ehat
        (m, i) = findmax(ehat ./ κ .^ slope)
        slopeconst = m
        dk = exp(log(kmax) * 0.5)
        # kpoints = κ[i] / dk, κ[i] * dk
        kpoints = κ[i] / (dk / 3), min(κ[i] * dk, kmax)
        slopepoints = @. slopeoffset * slopeconst * kpoints^slope
        [Point2f(kpoints[1], slopepoints[1]), Point2f(kpoints[2], slopepoints[2])]
    end

    # Nice ticks
    logmax = round(Int, log2(kmax + 1))
    xticks = T(2) .^ (0:logmax)

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xticks,
        xlabel = "k",
        # ylabel = "E(k)",
        xscale = log10,
        yscale = log10,
        limits = (1, kmax, T(1e-8), T(1)),
    )
    lines!(ax, κ, ehat; label = "Kinetic energy")
    lines!(ax, inertia; label = slopelabel, linestyle = :dash, color = Cycled(2))
    axislegend(ax; position = :lb)
    # autolimits!(ax)
    on(e -> autolimits!(ax), ehat)
    autolimits!(ax)
    fig
end

end
