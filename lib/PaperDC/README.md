# PaperDC

Scripts for generating results of the paper
[Discretize first, filter next: Learning divergence-consistent closure models for large-eddy simulation](https://www.sciencedirect.com/science/article/pii/S0021999124008258).

## Set up environment

Run:

```sh
julia --project=lib/PaperDC -e 'using Pkg; Pkg.instantiate()'
```

Now you can run the scripts in this directory. They generate results for

- `prioranalysis.jl`: Section 5.1 "Filtered DNS (2D and 3D)"
- `postanalysis.jl`: Section 5.2 "LES (2D)"
- `postanalysis3D.jl`: Appendix G.2. "LES of forced turbulence (3D)"
- `transferfunctions.jl`: Appendix E. "Continuous ﬁlters and transfer functions"
