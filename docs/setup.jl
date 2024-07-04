# Set up docs environment

using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(; path = joinpath(@__DIR__, "..")))
Pkg.develop(PackageSpec(; path = joinpath(@__DIR__, "..", "lib", "NeuralClosure")))
Pkg.develop(PackageSpec(; path = joinpath(@__DIR__, "..", "examples")))
Pkg.instantiate()
