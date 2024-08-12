# SciMLCompat

This repo makes IncompressibleNavierStokes compatible with SciML for the purpose of solving the Navier Stokes equation with `SciML.ODEProblem`.
In addition, we can also create neural closures and use Enzyme for AD.

The tests showcase how the implementations can be used.

## Set up environment

From this directory, run:

```sh
julia --project setup.jl
```

## Run the tests

Option 1:

```sh
julia --project test/runtests.jl
```

Option 2:

```sh
julia --project setup.jl
julia --project
]
test
```
