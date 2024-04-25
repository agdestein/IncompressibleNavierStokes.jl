# Examples

Example scripts for
[IncompressibleNavierStokes.jl](https://github.com/agdestein/IncompressibleNavierStokes.jl).

## Set up environment

From this directory, run:

```sh
julia --project -e '
using Pkg
Pkg.develop(PackageSpec(; path = ".."))
Pkg.instantiate()'
'
```

or interactively from a Julia REPL:

```julia-repl
julia> ]
(v1.10) pkg> activate .
(Examples) pkg> dev ..
(Examples) pkg> instantiate
```
