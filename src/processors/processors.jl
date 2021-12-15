"""
    Processor

Abstract iteration processor.
"""
abstract type Processor end

Base.@kwdef struct Logger <: Processor
    nupdate::Int = 1
end

Base.@kwdef mutable struct RealTimePlotter <: Processor
    nupdate::Int = 1
    fieldname::Symbol = :vorticity
    fps::Int = 60
    field::Node = Node(nothing)
end

Base.@kwdef mutable struct VTKWriter <: Processor
    nupdate::Int = 1
    dir::String = "output"
    filename::String = "solution"
    pvd::CollectionFile = paraview_collection("")
end

Base.@kwdef mutable struct QuantityTracer <: Processor
    nupdate::Int = 1
    t::Vector{Float64} = zeros(0)
    maxdiv::Vector{Float64} = zeros(0)
    umom::Vector{Float64} = zeros(0)
    vmom::Vector{Float64} = zeros(0)
    wmom::Vector{Float64} = zeros(0)
    k::Vector{Float64} = zeros(0)
end
