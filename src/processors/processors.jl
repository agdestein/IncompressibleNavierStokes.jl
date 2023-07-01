"""
    AbstractProcessor

Abstract iteration processor.
"""
abstract type AbstractProcessor end

"""
    Logger(nupdate)

Print time stepping information after every time step.
"""
Base.@kwdef struct Logger <: AbstractProcessor
    nupdate::Int = 1
end

"""
    StateObserver(nupdate, V, p, t)

Observe time, velocity and pressure field.

Let `o` be a `StateObserver`. Plotting `o.state`, or a quantity of interest
thereof, before solving an [`UnsteadyProblem`](@ref) with `o` as a processor,
results in a real time plot with a new frame every `nupdate`-th time step (when
the observable `o.state[] = (V, p, t)` is updated).

For example, to plot the total kinetic energy evolution, given the state
`V`, `p`, and `t`:

```julia
o = StateObserver(1, V, p, t)
_points = Point2f[]
points = @lift begin
    V, p, t = \$(o.state)
    E = sum(abs2, V)
    push!(_points, Point2f(t, E))
end
lines(points; axis = (; xlabel = "t", ylabel = "Kinetic energy"))
```

The plot is updated at every time step (`nupdate = 1`).
"""
struct StateObserver{T} <: AbstractProcessor
    nupdate::Int
    state::Observable{Tuple{Vector{T},Vector{T},T}}
end

StateObserver(nupdate, V, p, t) = StateObserver(nupdate, Observable((V, p, t)))

"""
    VTKWriter(; nupdate, dir = "output", filename = "solution")

Write the solution every `nupdate` time steps to a VTK file. The resulting Paraview data
collection file is stored in `"\$dir/\$filename.pvd"`.
"""
Base.@kwdef mutable struct VTKWriter <: AbstractProcessor
    nupdate::Int = 1
    dir::String = "output"
    filename::String = "solution"
    pvd::CollectionFile = paraview_collection("")
end
