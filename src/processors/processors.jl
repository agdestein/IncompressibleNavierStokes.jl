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
    RealTimePlotter(; nupdate, fieldname, type = heatmap)

Plot the solution every `nupdate` time steps. Available fieldnames are:

- `:velocity`,
- `:vorticity`,
- `:streamfunction`,
- `:pressure`.

Available plot `type`s are:

- `heatmap`,
- `contour`,
- `contourf`.
"""
Base.@kwdef mutable struct RealTimePlotter{F} <: AbstractProcessor
    nupdate::Int = 1
    fieldname::Symbol = :vorticity
    type::F = contour
    field::Observable = Observable(nothing)
    lims::Observable = Observable(nothing)
end

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

"""
    QuantityTracer(nupdate)

Store scalar quantities (maximum divergence, momentum, kinetic energy) every `nupdate` time
steps.
"""
Base.@kwdef mutable struct QuantityTracer <: AbstractProcessor
    nupdate::Int = 1
    t::Vector{Float64} = zeros(0)
    maxdiv::Vector{Float64} = zeros(0)
    umom::Vector{Float64} = zeros(0)
    vmom::Vector{Float64} = zeros(0)
    wmom::Vector{Float64} = zeros(0)
    k::Vector{Float64} = zeros(0)
end
