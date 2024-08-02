# Set up test environment for NeuralClosure

using Pkg

# Set up NeuralClosure
include(joinpath(@__DIR__, "..", "setup.jl"))

# Set up test env
Pkg.activate(@__DIR__)
Pkg.develop([
    # PackageSpec(; path = joinpath(@__DIR__, "..")),
    PackageSpec(; path = joinpath(@__DIR__, "..", "..", "..")),
])
Pkg.instantiate()
