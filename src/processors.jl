"""
Process results from time stepping. Before time stepping, the `initialize`
function is called on an observable of the time stepper `state`, returning
`initialized`. The observable is updated every time step.

After timestepping, the `finalize` function is called on `initialized` and the
final `state`.

See the following example:

```julia
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
Create processor that logs time step information.
"""
timelogger(;
    showiter = false,
    showt = true,
    showdt = true,
    showmax = true,
    showspeed = true,
    nupdate = 1,
) =
    processor() do state
        told = Ref(state[].t)
        oldtime = time()
        on(state) do (; u, t, n)
            Δt = t - told[]
            told[] = t
            n % nupdate == 0 || return
            newtime = time()
            itertime = (newtime - oldtime) / nupdate
            oldtime = newtime
            msg = String[]
            showiter && push!(msg, "Iteration $n")
            showt && push!(msg, @sprintf("t = %g", t))
            showdt && push!(msg, @sprintf("Δt = %.2g", Δt))
            showmax && push!(msg, @sprintf("umax = %.2g", maximum(abs, u)))
            showspeed && push!(msg, @sprintf("itertime = %.2g", itertime))
            @info join(msg, "\t")
        end
        nothing
    end

"""
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

    # Initialize buffers
    _f = if fieldname in (1, 2, 3)
        up = interpolate_u_p(u, setup)
        upf = selectdim(up, ndims(up), fieldname)
    elseif fieldname == :velocity
        up = interpolate_u_p(u, setup)
    elseif fieldname == :velocitynorm
        up = interpolate_u_p(u, setup)
        upnorm = scalarfield(setup)
    elseif fieldname == :vorticity
        ω = vorticity(u, setup)
        ωp = interpolate_ω_p(ω, setup)
    elseif fieldname == :streamfunction
        ψ = get_streamfunction(setup, u, t)
    elseif fieldname == :pressure
        if isnothing(psolver)
            @warn "Creating new pressure solver for observefield"
            psolver = default_psolver(setup)
        end
        F = vectorfield(setup)
        p = scalarfield(setup)
    elseif fieldname == :Dfield
        if isnothing(psolver)
            @warn "Creating new pressure solver for observefield"
            psolver = default_psolver(setup)
        end
        F = vectorfield(setup)
        p = scalarfield(setup)
        F = vectorfield(setup)
        d = scalarfield(setup)
    elseif fieldname == :Qfield
        Q = scalarfield(setup)
    elseif fieldname == :eig2field
        λ = scalarfield(setup)
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
    if ndims(_f) == D + 1
        _f = Array(_f)[Ip, :]
    elseif ndims(_f) == D
        _f = Array(_f)[Ip]
    else
        error()
    end

    # Observe field
    field = map(state) do (; u, temp, t)
        f = if fieldname in (1, 2, 3)
            interpolate_u_p!(up, u, setup)
            upf
        elseif fieldname == :velocity
            interpolate_u_p!(up, u, setup)
        elseif fieldname == :velocitynorm
            interpolate_u_p!(up, u, setup)
            # map((u, v, w) -> √sum(u^2 + v^2 + w^2), up...)
            if D == 2
                uptuple = eachslice(up; dims = ndims(up))
                @. upnorm = sqrt(uptuple[1]^2 + uptuple[2]^2)
            elseif D == 3
                uptuple = eachslice(up; dims = ndims(up))
                @. upnorm = sqrt(uptuple[1]^2 + uptuple[2]^2 + uptuple[3]^2)
            end
        elseif fieldname == :vorticity
            apply_bc_u!(u, t, setup)
            vorticity!(ω, u, setup)
            interpolate_ω_p!(ωp, ω, setup)
        elseif fieldname == :streamfunction
            get_streamfunction(setup, u, t)
        elseif fieldname == :pressure
            pressure!(p, u, temp, t, setup; psolver, F)
        elseif fieldname == :Dfield
            pressure!(p, u, temp, t, setup; psolver, F)
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
        if ndims(f) == D + 1
            copyto!(_f, view(f, Ip, :))
        elseif ndims(f) == D
            copyto!(_f, view(f, Ip))
        else
            error()
        end
        _f
    end
end

"""
In the case of a 2D setup, the velocity field is saved as a 3D vector with a
z-component of zero, as this seems to be preferred by ParaView.
"""
function snapshotsaver(state; setup, fieldnames = (:velocity,), psolver = nothing)
    state isa Observable || (state = Observable(state))
    (; grid) = setup
    (; dimension, xp, Ip) = grid
    D = dimension()
    xparr = getindex.(Array.(xp), Ip.indices)
    fields = map(fieldname -> observefield(state; setup, fieldname, psolver), fieldnames)

    # Only allocate z-component if there is a 2D vector field
    gz = if any(f -> ndims(f[]) == D + 1 && size(f[], D + 1) == 2, fields)
        zero(state[].u[Ip, 1])
    else
        nothing
    end

    savesnapshot!(filename, pvd = nothing) =
        vtk_grid(filename, xparr...) do vtk
            # Write fields to VTK file for current time
            for (fieldname, f) in zip(fieldnames, fields)
                # Exctract scalar channels fx, fy, fz
                g = f[]
                field = if ndims(g) == D
                    # Scalar field
                    g
                elseif ndims(g) == D + 1
                    # Vector field
                    if D == 2
                        # ParaView prefers 3D vectors. Add zero z-component.
                        g[:, :, 1], g[:, :, 2], gz
                    else
                        g[:, :, :, 1], g[:, :, :, 2], g[:, :, :, 3]
                    end
                end
                vtk[string(fieldname)] = field
            end

            # This is a special ParaView variable for non-uniform time stamp
            vtk["TimeValue"] = state[].t

            # Add VTK file for current time to collection file
            isnothing(pvd) || setindex!(pvd, vtk, state[].t)
        end
end

"""
Save fields to vtk file.

The `kwargs` are passed to [`snapshotsaver`](@ref).
"""
function save_vtk(state; setup, filename = "output/solution", kwargs...)
    path = dirname(filename)
    isdir(path) || mkpath(path)
    savesnapshot! = snapshotsaver(state; setup, kwargs...)
    savesnapshot!(filename)
end

"""
Create processor that writes the solution every `nupdate` time steps to a VTK
file. The resulting Paraview data collection file is stored in
`"\$dir/\$filename.pvd"`.
The `kwargs` are passed to [`snapshotsaver`](@ref).
"""
vtk_writer(; setup, nupdate = 1, dir = "output", filename = "solution", kwargs...) =
    processor((pvd, outerstate) -> vtk_save(pvd)) do outerstate
        ispath(dir) || mkpath(dir)
        pvd = paraview_collection(joinpath(dir, filename))
        state = Observable(outerstate[])
        savesnapshot! = snapshotsaver(state; setup, kwargs...)

        # Update VTK file
        on(outerstate) do outerstate
            (; t, n) = outerstate
            n % nupdate == 0 || return
            state[] = outerstate
            tformat = replace(string(t), "." => "p")
            savesnapshot!("$(dir)/$(filename)_t=$tformat", pvd)
        end

        # Initial step
        outerstate[] = outerstate[]
        pvd
    end

"""
Create processor that stores the solution and time every `nupdate` time step.
"""
fieldsaver(; setup, nupdate = 1) =
    processor() do state
        states = fill(adapt(Array, state[]), 0)
        on(state) do state
            state.n % nupdate == 0 || return
            state = adapt(Array, state)
            state.u isa Array && (state = deepcopy(state))
            push!(states, state)
        end
        states
    end

"Observe energy spectrum of `state`."
function observespectrum(state; setup, npoint = 100, a = typeof(setup.Re)(1 + sqrt(5)) / 2)
    state isa Observable || (state = Observable(state))

    (; dimension, xp, Ip, Np) = setup.grid
    T = eltype(xp[1])
    D = dimension()

    (; inds, κ, K) = spectral_stuff(setup; npoint, a)

    # Energy
    uhat = similar(xp[1], Complex{T}, Np)
    # up = interpolate_u_p(state[].u, setup)
    _ehat = zeros(T, length(κ))
    ehat = map(state) do (; u)
        # interpolate_u_p!(up, u, setup)
        up = u
        # TODO: Maybe preallocate e and A * e
        e = sum(eachslice(up; dims = D + 1)) do u
            copyto!(uhat, view(u, Ip))
            fft!(uhat)
            uhathalf = view(uhat, ntuple(α -> 1:K[α], D)...)
            abs2.(uhathalf) ./ (2 * prod(Np)^2)
        end
        e = map(i -> sum(view(e, i)), inds)
        # e = max.(e, eps(T)) # Avoid log(0)
        copyto!(_ehat, e)
    end

    (; ehat, κ)
end

# These empty functions are defined here, but implemented in
# ext/IncompressibleNavierStokesMakieExt.jl
# To use them, load Makie using e.g. GLMakie or CairoMakie
# This reduces a lot of dependencies if plotting is not required
function animator end
function realtimeplotter end
function fieldplot end
function energy_history_plot end
function energy_spectrum_plot end

# Add docstrings here, otherwise Documenter can't find them

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
animator

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
realtimeplotter

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
fieldplot

"Create energy history plot."
energy_history_plot

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
energy_spectrum_plot
