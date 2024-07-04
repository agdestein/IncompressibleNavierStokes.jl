# Set up PaperDC environment

using Pkg

Pkg.activate(@__DIR__)
Pkg.develop([
    PackageSpec(; path = joinpath(@__DIR__, "..", "..")),
    PackageSpec(; path = joinpath(@__DIR__, "..", "NeuralClosure")),
])
Pkg.instantiate()
