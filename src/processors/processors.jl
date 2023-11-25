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
timelogger(; nupdate = 1) = processor(function (state)
    on(state) do (; t, n)
        n % nupdate == 0 || return
        @printf "Iteration %d\tt = %g\n" n t
    end
    nothing
end)

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
) =
    processor((pvd, state) -> vtk_save(pvd)) do state
        (; grid) = setup
        (; dimension, xp) = grid
        D = dimension()
        ispath(dir) || mkpath(dir)
        pvd = paraview_collection(joinpath(dir, filename))
        xp = Array.(xp)
        on(state) do (; u, p, t, n)
            n % nupdate == 0 || return
            tformat = replace(string(t), "." => "p")
            vtk_grid("$(dir)/$(filename)_t=$tformat", xp...) do vtk
                if :velocity ∈ fields
                    up = interpolate_u_p(u, setup)
                    if D == 2
                        # ParaView prefers 3D vectors. Add zero z-component.
                        up = (up..., zero(up[1]))
                    end
                    vtk["velocity"] = Array.(up)
                end
                :pressure ∈ fields && (vtk["pressure"] = Array(p))
                if :vorticity ∈ fields
                    ω = interpolate_ω_p(vorticity(u, setup), setup)
                    ω = D == 2 ? Array(ω) : Array.(ω)
                    vtk["vorticity"] = ω
                end
                pvd[t] = vtk
            end
        end
        pvd
    end

"""
    fieldsaver(; setup, nupdate = 1)

Create processor that stores the solution every `nupdate` time step to the vector of vectors `V` and `p`. The solution times are stored in the vector `t`.
"""
field_saver(; setup, nupdate = 1) =
    processor() do state
        T = eltype(setup.grid.x[1])
        (; u, p) = state[]
        fields = (; u = fill(Array.(u), 0), p = fill(Array(p), 0), t = zeros(T, 0))
        on(state) do (; u, p, t, n)
            n % nupdate == 0 || return
            push!(fields.u, Array.(u))
            push!(fields.p, Array(p))
            push!(fields.t, t)
        end
        fields
    end
