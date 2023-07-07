raw"""
    processor(
        initialize;
        finalize = (initialized, stepper) -> initialized,
        nupdate = 1
    )

Process results from time stepping. Before time stepping, the `initialize` function is called on an observable of the time `stepper`, returning `initialized`. The observable is updated after every `nupdate` time step, triggering updates where `@lift` is used inside `initialize`.

After timestepping, the `finalize` function is called on `initialized` and the final stepper.

See the following example:

```example
function initialize(step_observer)
    s = 0
    println("Let's sum up the time steps")
    @lift begin
        (; n) = $step_observer
        println("The summand is $n")
        s = s + n
    end
    s
end

finalize(s, stepper) = println("The final sum (at time t=$(stepper.t)) is $s")
p = Processor(intialize; finalize, nupdate = 5)
```

When solved for 20 time steps from t=0 to t=2 the displayed output is

```
Let's sum up the time steps
The summand is 0
The summand is 5
The summand is 10
The summand is 15
The summand is 20
The final sum (at time t=2.0) is 50
```
"""
processor(initialize; finalize = (initialized, stepper) -> initialized, nupdate = 1) =
    (; initialize, finalize, nupdate)

"""
    step_logger(; nupdate = 1)

Create processor that logs time step information.
"""
step_logger(; nupdate = 1) = processor(
    function (step_observer) 
        @lift begin
            (; t, n) = $step_observer
            @printf "Iteration %d\tt = %g\n" n t
        end
        nothing
    end;
    nupdate,
)

"""
    vtk_writer(setup; nupdate, dir = "output", filename = "solution")

Create processor that writes the solution every `nupdate` time steps to a VTK file. The resulting Paraview data
collection file is stored in `"\$dir/\$filename.pvd"`.
"""
vtk_writer(setup; nupdate = 1, dir = "output", filename = "solution") = processor(
    function (step_observer)
        ispath(dir) || mkpath(dir)
        pvd = paraview_collection(joinpath(dir, filename))
        @lift begin
            (; grid) = setup
            (; dimension, xp, yp) = grid
            (; V, p, t) = $step_observer

            V = Array(V)
            p = Array(p)

            N = dimension()
            if N == 2
                coords = (xp, yp)
            elseif N == 3
                (; zp) = grid
                coords = (xp, yp, zp)
            end

            tformat = replace(string(t), "." => "p")
            vtk_grid("$(dir)/$(filename)_t=$tformat", coords...) do vtk
                vels = get_velocity(setup, V, t)
                if N == 2
                    # ParaView prefers 3D vectors. Add zero z-component.
                    wp = zeros(size(vels[1]))
                    vels = (vels..., wp)
                end
                vtk["velocity"] = vels
                vtk["pressure"] = p
                pvd[t] = vtk
            end
        end
        pvd
    end;
    finalize = (pvd, step_observer) -> vtk_save(pvd),
    nupdate,
)

"""
    field_saver(setup; nupdate = 1)

Create processor that stores the solution every `nupdate` time step to the vector of vectors `V` and `p`. The solution times are stored in the vector `t`.
"""
field_saver(setup; nupdate = 1) = processor(function (step_observer)
    T = eltype(setup.grid.x)
    _V = fill(zeros(T, 0), 0)
    _p = fill(zeros(T, 0), 0)
    _t = fill(zero(T), 0)
    @lift begin
        (; V, p, t) = $step_observer
        push!(_V, Array(V))
        push!(_p, Array(p))
        push!(_t, t)
    end
    (; V = _V, p = _p, t = _t)
end; nupdate)
