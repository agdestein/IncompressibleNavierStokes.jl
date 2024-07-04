# Set up test environment for IncompressibleNavierStokes

using Pkg

Pkg.activate(@__DIR__)
# Pkg.develop(PackageSpec(; path = joinpath(@__DIR__, "..")))
Pkg.instantiate()
