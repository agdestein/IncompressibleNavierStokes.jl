"""
    processor(initialize, finalize = (initialized, state) -> initialized)

Process results from time stepping. Before time stepping, the `initialize`
function is called on an observable of the time stepper `state`, returning
`initialized`. The observable is updated every time step.

After timestepping, the `finalize` function is called on `initialized` and the
final `state`.

See the following example:

```example
function initialize(state)
    s = 0
    println("Let's sum up the time steps")
    on(state) do (; n, t)
        println("The summand is \$n, the time is \$t")
        s = s + n
    end
    s
end

finalize(i, state) = println("The final sum (at time t=\$(state.t)) is \$s")
p = processor(initialize, finalize)
```

When solved for 6 time steps from t=0 to t=2 the displayed output is

```
Let's sum up the time steps
The summand is 0, the time is 0.0
The summand is 1, the time is 0.4
The summand is 2, the time is 0.8
The summand is 3, the time is 1.2
The summand is 4, the time is 1.6
The summand is 5, the time is 2.0
The final sum (at time t=2.0) is 15
```
"""
processor(initialize, finalize = (initialized, state) -> initialized) =
    (; initialize, finalize)

"""
    timelogger(; showmax = true, nupdate = 1)

Create processor that logs time step information.
"""
timelogger(; showmax = true, nupdate = 1) =
    processor() do state
        on(state) do (; u, t, n)
            n % nupdate == 0 || return
            @printf "Iteration %d\tt = %g" n t
            showmax && @printf "\tumax = %g" maximum(maximum.(abs, u))
            println()
        end
        nothing
    end

"""
    observefield(
        state;
        setup,
        fieldname,
        logtol = eps(eltype(setup.grid.x[1])),
        psolver = nothing,
    )

Observe field `fieldname` at pressure points.
"""
function observefield(
    state;
    setup,
    fieldname,
    logtol = eps(eltype(setup.grid.x[1])),
    psolver = nothing,
)
    (; dimension, Ip) = setup.grid
    (; u, temp, t) = state[]
    D = dimension()
    T = eltype(u[1])

    # Initialize buffers
    _f = if fieldname in (1, 2, 3)
        up = interpolate_u_p(u, setup)
        up[fieldname]
    elseif fieldname == :velocity
        up = interpolate_u_p(u, setup)
    elseif fieldname == :velocitynorm
        up = interpolate_u_p(u, setup)
        upnorm = zero(up[1])
    elseif fieldname == :vorticity
        ω = vorticity(u, setup)
        ωp = interpolate_ω_p(ω, setup)
    elseif fieldname == :streamfunction
        ψ = get_streamfunction(setup, u, t)
    elseif fieldname == :pressure
        if isnothing(psolver)
            @warn "Creating new pressure solver for fieldplot"
            psolver = default_psolver(setup)
        end
        F = zero.(u)
        div = zero(u[1])
        p = zero(u[1])
    elseif fieldname == :Dfield
        if isnothing(psolver)
            @warn "Creating new pressure solver for fieldplot"
            psolver = default_psolver(setup)
        end
        F = zero.(u)
        div = zero(u[1])
        p = zero(u[1])
        G = similar.(u)
        d = similar(u[1])
    elseif fieldname == :Qfield
        Q = similar(u[1])
    elseif fieldname == :eig2field
        λ = similar(u[1])
    elseif fieldname in union(Symbol.(["B$i" for i = 1:11]), Symbol.(["V$i" for i = 1:5]))
        sym = string(fieldname)[1]
        sym = sym == 'B' ? 1 : 2
        idx = parse(Int, string(fieldname)[2:end])
        tb = tensorbasis(u, setup)
        tb[sym][idx]
    elseif fieldname == :temperature
        temp
    else
        error("Unknown fieldname")
    end
    _f = _f isa Tuple ? map(f -> Array(f)[Ip], _f) : Array(_f)[Ip]

    # Observe field
    field = lift(state) do (; u, temp, t)
        f = if fieldname in (1, 2, 3)
            interpolate_u_p!(up, u, setup)
            up[fieldname]
        elseif fieldname == :velocity
            interpolate_u_p!(up, u, setup)
        elseif fieldname == :velocitynorm
            interpolate_u_p!(up, u, setup)
            # map((u, v, w) -> √sum(u^2 + v^2 + w^2), up...)
            if D == 2
                @. upnorm = sqrt(up[1]^2 + up[2]^2)
            elseif D == 3
                @. upnorm = sqrt(up[1]^2 + up[2]^2 + up[3]^2)
            end
        elseif fieldname == :vorticity
            apply_bc_u!(u, t, setup)
            vorticity!(ω, u, setup)
            interpolate_ω_p!(ωp, ω, setup)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, u, t)
        elseif fieldname == :pressure
            pressure!(p, u, temp, t, setup; psolver, F, div)
        elseif fieldname == :Dfield
            pressure!(p, u, temp, t, setup; psolver, F, div)
            Dfield!(d, G, p, setup)
            din = view(d, Ip)
            @. din = log(max(logtol, din))
            d
        elseif fieldname == :Qfield
            Qfield!(Q, u, setup)
            Qin = view(Q, Ip)
            @. Qin = log(max(logtol, Qin))
            Q
        elseif fieldname == :eig2field
            eig2field!(λ, u, setup)
            λin = view(λ, Ip)
            @. λin .= log(max(logtol, -λin))
            λ
        elseif fieldname in
               union(Symbol.(["B$i" for i = 1:11]), Symbol.(["V$i" for i = 1:5]))
            tensorbasis!(tb..., u, setup)
            tb[sym][idx]
        elseif fieldname == :temperature
            temp
        end
        if _f isa Tuple
            for (_f, f) in zip(_f, f)
                copyto!(_f, view(f, Ip))
            end
        else
            copyto!(_f, view(f, Ip))
        end
        _f
    end
end

"""
    vtk_writer(;
        setup,
        nupdate = 1,
        dir = "output",
        filename = "solution",
        fields = (:velocity,),
    )

Create processor that writes the solution every `nupdate` time steps to a VTK
file. The resulting Paraview data collection file is stored in
`"\$dir/\$filename.pvd"`.
"""
vtk_writer(;
    setup,
    nupdate = 1,
    dir = "output",
    filename = "solution",
    fieldnames = (:velocity,),
    psolver = nothing,
) =
    processor((pvd, outerstate) -> vtk_save(pvd)) do outerstate
        (; grid) = setup
        (; dimension, xp, Ip) = grid
        D = dimension()
        ispath(dir) || mkpath(dir)
        pvd = paraview_collection(joinpath(dir, filename))
        xparr = getindex.(Array.(xp), Ip.indices)
        state = Observable(outerstate[])
        f = map(fieldname -> observefield(state; setup, fieldname, psolver), fieldnames)

        # Only allocate z-component if there is a 2D vector field
        z = if any(f -> f[] isa Tuple && length(f[]) == 2, f)
            zero(state[].u[1][Ip])
        else
            nothing
        end

        # Update VTK file
        on(outerstate) do outerstate
            (; t, n) = outerstate
            n % nupdate == 0 || return
            state[] = outerstate
            tformat = replace(string(t), "." => "p")
            vtk_grid("$(dir)/$(filename)_t=$tformat", xparr...) do vtk
                for (fieldname, f) in zip(fieldnames, f)
                    field = if f[] isa Tuple && length(f[]) == 2
                        # ParaView prefers 3D vectors. Add zero z-component.
                        (f[]..., z)
                    else
                        f[]
                    end
                    vtk[string(fieldname)] = field
                end
                pvd[t] = vtk
            end
        end
        # Initial step
        outerstate[] = outerstate[]
        pvd
    end

"""
    fieldsaver(; setup, nupdate = 1)

Create processor that stores the solution and time every `nupdate` time step.
"""
fieldsaver(; setup, nupdate = 1) =
    processor() do state
        T = eltype(setup.grid.x[1])
        (; u) = state[]
        fields = (; u = fill(Array.(u), 0), t = zeros(T, 0))
        on(state) do (; u, p, t, n)
            n % nupdate == 0 || return
            push!(fields.u, Array.(u))
            push!(fields.t, t)
        end
        fields
    end

"""
    animator(; setup, path, plot = fieldplot, nupdate = 1, kwargs...)

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
    kwargs...,
) =
    processor((stream, state) -> save(path, stream)) do outerstate
        ispath(dirname(path)) || mkpath(dirname(path))
        state = Observable(outerstate[])
        fig = plot(state; setup, kwargs...)
        visible && display(fig)
        stream = VideoStream(fig; framerate, visible)
        on(outerstate) do outerstate
            outerstate.n % nupdate == 0 || return
            state[] = outerstate
            recordframe!(stream)
        end
        stream
    end

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
    kwargs...,
)
    (; grid) = setup
    (; xp, Ip) = grid

    xf = Array.(getindex.(xp, Ip.indices))
    dxf = diff.(xf)
    if all(α -> all(≈(dxf[α][1]), dxf[α]), 1:3)
        xf = ntuple(α -> LinRange(xf[α][1], xf[α][end], length(xf[α])), 3)
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
    docolorbar && Colorbar(fig[1, 2], hm)
    fig
end

"""
    energy_history_plot(state; setup)

Create energy history plot.
"""
function energy_history_plot(state; setup)
    @assert state isa Observable "Energy history requires observable state."
    (; Ω, Ip) = setup.grid
    e = zero(state[].u[1])
    _points = Point2f[]
    points = lift(state) do (; u, t)
        kinetic_energy!(e, u, setup)
        e .*= Ω
        E = sum(e[Ip])
        push!(_points, Point2f(t, E))
    end
    fig = lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))
    on(_ -> autolimits!(fig.axis), points)
    fig
end

"""
    energy_spectrum_plot(state; setup)

Create energy spectrum plot.
The energy at a scalar wavenumber level ``\\kappa \\in \\mathbb{N}`` is defined by

```math
\\hat{e}(\\kappa) = \\int_{\\kappa \\leq \\| k \\|_2 < \\kappa + 1} | \\hat{e}(k) | \\mathrm{d} k,
```

as in San and Staples [San2012](@cite).
"""
function energy_spectrum_plot(
    state;
    setup,
    npoint = 100,
    a = typeof(setup.Re)(1 + sqrt(5)) / 2,
)
    state isa Observable || (state = Observable(state))

    (; dimension, xp, Ip) = setup.grid
    T = eltype(xp[1])
    D = dimension()

    (; A, κ, K) = spectral_stuff(setup; npoint, a)
    # (; masks, κ, K) = get_spectrum(setup; npoint, a) # Mask
    kmax = maximum(κ)

    # Energy
    # up = interpolate_u_p(state[].u, setup)
    ehat = lift(state) do (; u, t)
        # interpolate_u_p!(up, u, setup)
        up = u
        e = sum(up) do u
            u = u[Ip]
            uhat = fft(u)[ntuple(α -> 1:K[α], D)...]
            # uhat = fft(u)[ntuple(α -> 1:K, D)...] # Mask
            abs2.(uhat) ./ (2 * prod(size(uhat))^2)
        end
        e = A * reshape(e, :)
        # e = [sum(e[m]) for m in masks] # Mask
        e = max.(e, eps(T)) # Avoid log(0)
        Array(e)
    end

    # Build inertial slope above energy
    # krange = LinRange(1, kmax, 100)
    # krange = collect(1, kmax)
    # krange = [cbrt(T(kmax)), T(kmax)]
    krange = [kmax^T(0.3), kmax^(T(0.8))]
    # krange = [T(kmax)^(T(2) / 3), T(kmax)]
    slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}$") : (-T(5 / 3), L"$k^{-5/3}$")
    inertia = lift(ehat) do ehat
        slopeconst = maximum(ehat ./ κ .^ slope)
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
        # ylabel = "E(k)",
        xscale = log10,
        yscale = log10,
        limits = (1, kmax, T(1e-8), T(1)),
    )
    lines!(ax, κ, ehat; label = "Kinetic energy")
    lines!(ax, krange, inertia; label = slopelabel, linestyle = :dash)
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
#     on(render.render_tick) do _
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
