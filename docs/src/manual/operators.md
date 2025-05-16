```@meta
CurrentModule = IncompressibleNavierStokes
```

# Operators

All discrete operators are built using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/)
and Cartesian indices, similar to
[WaterLily.jl](https://github.com/weymouth/WaterLily.jl/).
This allows for dimension- and backend-agnostic code. See this
[blog post](https://b-fg.github.io/research/2023-07-05-waterlily-on-gpu.html)
for how to write kernels. IncompressibleNavierStokes previously relied on
assembling sparse operators to perform the same operations. While being very
efficient and also compatible with CUDA (CUSPARSE), storing these matrices in
memory is expensive for large 3D problems.

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["operators.jl", "tensorbasis.jl"]
```
