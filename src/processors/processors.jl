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
    timelogger(; nupdate = 1)

Create processor that logs time step information.
"""
timelogger(; nupdate = 1) =
    processor() do state
        on(state) do (; t, n)
            n % nupdate == 0 || return
            @printf "Iteration %d\tt = %g\n" n t
        end
        nothing
    end

"""
    vtk_writer(;
        setup,
        nupdate = 1,
        dir = "output",
        filename = "solution",
        fields = (:velocity, :pressure, :vorticity),
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
    fields = (:velocity, :pressure, :vorticity),
    psolver = nothing,
) =
    processor((pvd, state) -> vtk_save(pvd)) do state
        (; grid) = setup
        (; dimension, xp) = grid
        D = dimension()
        ispath(dir) || mkpath(dir)
        pvd = paraview_collection(joinpath(dir, filename))
        xparr = Array.(xp)
        (; u) = state[]
        if :velocity ∈ fields
            up = interpolate_u_p(u, setup)
            uparr = Array.(up)
            # ParaView prefers 3D vectors. Add zero z-component.
            D == 2 && (uparr = (uparr..., zero(up[1])))
        end
        if :pressure ∈ fields
            if isnothing(psolver)
                @info "Creating new pressure solver for vtk_writer"
                psolver = psolver_direct(setup)
            end
            F = zero.(u)
            div = zero(u[1])
            p = zero(u[1])
            parr = adapt(Array, p)
        end
        if :vorticity ∈ fields
            ω = vorticity(u, setup)
            ωp = interpolate_ω_p(ω, setup)
            ωparr = adapt(Array, ωp)
        end
        on(state) do (; u, t, n)
            n % nupdate == 0 || return
            tformat = replace(string(t), "." => "p")
            vtk_grid("$(dir)/$(filename)_t=$tformat", xparr...) do vtk
                if :velocity ∈ fields
                    interpolate_u_p!(up, u, setup)
                    copyto!.(uparr, up)
                    vtk["velocity"] = uparr
                end
                if :pressure ∈ fields
                    pressure!(p, u, setup; psolver, F, div)
                    vtk["pressure"] = copyto!(parr, p)
                end
                if :vorticity ∈ fields
                    vorticity!(ω, u, setup)
                    interpolate_ω_p!(ωp, ω, setup)
                    D == 2 ? copyto!(ωparr, ωp) : copyto!.(ωparr, ωp)
                    vtk["vorticity"] = ωparr
                end
                pvd[t] = vtk
            end
        end
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
