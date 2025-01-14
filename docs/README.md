# Docs

Documentation for
[IncompressibleNavierStokes.jl](https://github.com/agdestein/IncompressibleNavierStokes.jl).

## Build documentation

To build the documentation locally, run:

```sh
julia --project=docs -e "using Pkg; Pkg.instantiate()"
```

and then run

```sh
julia --project=docs docs/make.jl
```
