# Docs

Documentation for
[IncompressibleNavierStokes.jl](https://github.com/agdestein/IncompressibleNavierStokes.jl).

## Set up environment

From this directory, run:

```sh
julia setup.jl
```

## Build documentation

To build the documentation locally, run:

```sh
julia --project=docs docs/make.jl
```

## Build documentation with live preview

Add a [`LiveServer.jl`](https://github.com/tlienart/LiveServer.jl) environment:

```sh
julia --project=@LiveServer -e 'using Pkg; Pkg.add("LiveServer")'
```

Then run:

```sh
julia .tools/servedocs.jl
```
