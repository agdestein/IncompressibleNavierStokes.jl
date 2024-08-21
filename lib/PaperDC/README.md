# PaperDC

Scripts for generating results of the paper
[Discretize first, filter next: learning divergence-consistent closure models for large-eddy simulation](https://arxiv.org/abs/2403.18088).

## Set up environment

Run:

```sh
julia --project=lib/PaperDC -e 'using Pkg; Pkg.instantiate()'
```

Now you can run the scripts in this directory:

- `prioranalysis.jl`: Generate results for section 5.1 "Filtered DNS (2D and 3D)"
- `postanalysis.jl`: Generate results for section 5.2 "LES (2D)"
