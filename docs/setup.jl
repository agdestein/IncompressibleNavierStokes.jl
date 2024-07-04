# Set up docs environment

using Pkg

Pkg.activate(@__DIR__)
Pkg.develop([
    PackageSpec(; path = joinpath(@__DIR__, "..")),
    PackageSpec(; path = joinpath(@__DIR__, "..", "lib", "NeuralClosure")),
    PackageSpec(; path = joinpath(@__DIR__, "..", "examples")),
])
Pkg.instantiate()
