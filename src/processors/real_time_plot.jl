"""
    realtimeplotter(;
        setup,
        plot = fieldplot,
        nupdate = 1,
        displayfig = true,
        displayupdates = false,
        sleeptime = nothing,
        kwargs...,
    )

Processor for plotting the solution every `nupdate` time step.

The `sleeptime` is slept at every update, to give Makie time to update the
plot. Set this to `nothing` to skip sleeping.

Additional `kwargs` are passed to the `plot` function.
"""
realtimeplotter(;
    setup,
    plot = fieldplot,
    nupdate = 1,
    displayfig = true,
    displayupdates = false,
    sleeptime = nothing,
    kwargs...,
) =
    processor() do outerstate
        state = Observable(outerstate[])
        fig = plot(state; setup, kwargs...)
        displayfig && display(fig)
        on(outerstate) do outerstate
            outerstate.n % nupdate == 0 || return
            state[] = outerstate
            displayupdates && display(fig)
            isnothing(sleeptime) || sleep(sleeptime)
        end
        fig
    end

"""
    fieldplot(
        state;
        setup,
        fieldname = :vorticity,
        type = nothing,
        kwargs...,
    )

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
    type = heatmap,
    equal_axis = true,
    docolorbar = true,
    size = nothing,
    kwargs...,
)
    (; boundary_conditions, grid) = setup
    (; dimension, xlims, x, xp, Ip) = grid
    D = dimension()

    xf = Array.(getindex.(setup.grid.xp, Ip.indices))

    (; u, p, t) = state[]
    _f = if fieldname in (1, 2)
        up = interpolate_u_p(u, setup)
        up[fieldname]
    elseif fieldname == :velocity
        up = interpolate_u_p(u, setup)
        upnorm = zero(p)
    elseif fieldname == :vorticity
        ω = vorticity(u, setup)
        ωp = interpolate_ω_p(ω, setup)
    elseif fieldname == :streamfunction
        ψ = get_streamfunction(setup, u, t)
    elseif fieldname == :pressure
        p
    end
    _f = Array(_f)[Ip]
    field = lift(state) do (; u, p, t)
        f = if fieldname in (1, 2)
            interpolate_u_p!(up, u, setup)
            up[fieldname]
        elseif fieldname == :velocity
            interpolate_u_p!(up, u, setup)
            map((u, v) -> √sum(u^2 + v^2), up...)
            @. upnorm = sqrt(up[1]^2 + up[2]^2)
        elseif fieldname == :vorticity
            apply_bc_u!(u, t, setup)
            vorticity!(ω, u, setup)
            interpolate_ω_p!(ωp, ω, setup)
        elseif fieldname == :streamfunction
            get_streamfunction!(setup, ψ, u, t)
        elseif fieldname == :pressure
            p
        end
        # Array(f)[Ip]
        copyto!(_f, view(f, Ip))
    end

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
            colorrange = lims,
            kwargs...,
        )
    end

    axis = (;
        title = titlecase(string(fieldname)),
        xlabel = "x",
        ylabel = "y",
        limits = (xlims[1]..., xlims[2]...),
    )
    equal_axis && (axis = (axis..., aspect = DataAspect()))

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
    fieldname = :eig2field,
    alpha = convert(eltype(setup.grid.x[1]), 0.1),
    isorange = convert(eltype(setup.grid.x[1]), 0.5),
    equal_axis = true,
    levels = 3,
    docolorbar = false,
    size = nothing,
    kwargs...,
)
    (; boundary_conditions, grid) = setup
    (; xlims, x, xp, Ip) = grid

    xf = Array.(getindex.(setup.grid.xp, Ip.indices))
    (; u, p) = state[]
    if fieldname == :velocity
    elseif fieldname == :vorticity
    elseif fieldname == :streamfunction
    elseif fieldname == :pressure
    elseif fieldname == :Dfield
        G = similar.(state[].u)
        d = similar(state[].p)
    elseif fieldname == :Qfield
        Q = similar(state[].p)
    elseif fieldname == :eig2field
        λ = similar(state[].p)
    else
        error("Unknown fieldname")
    end

    field = lift(state) do (; u, p, t)
        f = if fieldname == :velocity
            up = interpolate_u_p(u, setup)
            map((u, v, w) -> √sum(u^2 + v^2 + w^2), up...)
        elseif fieldname == :vorticity
            ωp = interpolate_ω_p(vorticity(u, setup), setup)
            map((u, v, w) -> √sum(u^2 + v^2 + w^2), ωp...)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, u, t)
        elseif fieldname == :pressure
            p
        elseif fieldname == :Dfield
            Dfield!(d, G, p, setup)
            d
        elseif fieldname == :Qfield
            Qfield!(Q, u, setup)
            Q
        elseif fieldname == :eig2field
            eig2field!(λ, u, setup)
            λ
        end
        Array(f)[Ip]
    end

    # lims = @lift get_lims($field)
    lims = isnothing(levels) ? lift(get_lims, field) : extrema(levels)

    isnothing(levels) && (levels = @lift(LinRange($(lims)..., 10)))

    # aspect = equal_axis ? (; aspect = :data) : (;)
    size = isnothing(size) ? (;) : (; size)
    fig = Figure(; size...)
    # ax = Axis3(fig[1, 1]; title = titlecase(string(fieldname)), aspect...)
    hm = contour(
        fig[1, 1],
        # ax,
        xf...,
        field;
        levels,
        colorrange = lims,
        # colorrange = extrema(levels),
        alpha,
        isorange,
        # highclip = :red,
        # lowclip = :red,
        kwargs...,
    )

    docolorbar && Colorbar(fig[1, 2], hm)
    fig
end

"""
    energy_history_plot(state; setup)

Create energy history plot.
"""
function energy_history_plot(state; setup)
    @assert state isa Observable "Energy history requires observable state."
    _points = Point2f[]
    points = lift(state) do (; u, p, t)
        E = kinetic_energy(u, setup)
        push!(_points, Point2f(t, E))
    end
    fig = lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))
    on(_ -> autolimits!(fig.axis), points)
    fig
end

"""
    energy_spectrum_plot(state; setup, doaverage = false)

Create energy spectrum plot.
The energy at a scalar wavenumber level ``\\kappa \\in \\mathbb{N}`` is defined by

```math
\\hat{e}(\\kappa) = \\int_{\\kappa \\leq \\| k \\|_2 < \\kappa + 1} | \\hat{e}(k) | \\mathrm{d} k,
```

as in San and Staples [San2012](@cite).
"""
function energy_spectrum_plot(state; setup)
    state isa Observable || (state = Observable(state))
    (; dimension, xp, Ip) = setup.grid
    backend = get_backend(xp[1])
    T = eltype(xp[1])
    D = dimension()
    K = size(Ip) .÷ 2
    kx = ntuple(α -> 0:K[α]-1, D)
    k = fill!(similar(xp[1], length.(kx)), 0)
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)

    # Sum or average wavenumbers between k and k+1
    kmax = minimum(K) - 1
    nk = ceil(Int, maximum(k))
    kint = 1:kmax
    ia = similar(xp[1], Int, 0)
    ib = sortperm(k)
    vals = similar(xp[1], 0)
    ksort = k[ib]
    jprev = 2 # Do not include constant mode
    for ki = 1:kmax
        j = findfirst(>(ki+1), ksort)
        isnothing(j) && (j = length(k) + 1)
        ia = [ia; fill!(similar(ia, j - jprev), ki)]
        # val = doaverage ? T(1) / (j - jprev) : T(1)
        val = T(π) * ((ki+1)^2 - ki^2) / (j - jprev)
        vals = [vals; fill!(similar(vals, j - jprev), val)]
        jprev = j
    end
    ib = ib[2:jprev-1]
    A = sparse(ia, ib, vals, kmax, length(k))

    # Energy
    up = interpolate_u_p(state[].u, setup)
    ehat = lift(state) do (; u, p, t)
        interpolate_u_p!(up, u, setup)
        e = sum(up -> up[Ip] .^ 2, up)
        e = fft(e)[ntuple(α -> kx[α] .+ 1, D)...]
        e = abs.(e) ./ size(e, 1)
        e = A * reshape(e, :)
        e = max.(e, eps(T)) # Avoid log(0)
        Array(e)
    end

    # Build inertial slope above energy
    # krange = LinRange(extrema(kint)..., 100)
    # krange = collect(extrema(kint))
    krange = [cbrt(T(kmax)), T(kmax)]
    slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}") : (-T(5 / 3), L"$k^{-5/3}")
    inertia = lift(ehat) do ehat
        slopeconst = maximum(ehat ./ kint .^ slope)
        2 .* slopeconst .* krange .^ slope
    end

    # Nice ticks
    logmax = round(Int, log2(kmax + 1))
    xticks = T(2) .^ (0:logmax)

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xticks,
        xlabel = "k",
        ylabel = "e(k)",
        xscale = log10,
        yscale = log10,
        limits = (extrema(kint)..., T(1e-8), T(1)),
    )
    lines!(ax, kint, ehat; label = "Kinetic energy")
    lines!(ax, krange, inertia; label = slopelabel, color = :red)
    axislegend(ax)
    # autolimits!(ax)
    on(e -> autolimits!(ax), ehat)
    autolimits!(ax)
    fig
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
