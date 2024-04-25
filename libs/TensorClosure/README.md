# TensorClosure

Tensor closure scripts.

## Set up environment

From this directory, run:

```sh
julia --project -e '
using Pkg
Pkg.develop([
    PackageSpec(; path = "../.."),
    PackageSpec(; path = "../NeuralClosure"),
])
Pkg.instantiate()'
'
```

or interactively from a Julia REPL:

```julia-repl
julia> ]
(v1.10) pkg> activate .
(TensorClosure) pkg> dev ../.. ../NeuralClosure
(TensorClosure) pkg> instantiate
```
