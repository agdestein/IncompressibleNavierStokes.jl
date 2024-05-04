```@meta
CurrentModule = IncompressibleNavierStokes
```

# Operators

All discrete operators are built using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/)
and Cartesian indices, similar to
[WaterLily.jl](https://github.com/weymouth/WaterLily.jl/).
This allows for dimension- and backend-agnostic code. See this
[blog post](https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html)
for how to write kernels. IncompressibleNavierStokes previously relied on
assembling sparse operators to perform the same operations. While being very
efficient and also compatible with CUDA (CUSPARSE), storing these matrices in
memory is expensive for large 3D problems.

```@docs
Offset
divergence!
divergence
vorticity
vorticity!
convection!
diffusion!
bodyforce!
momentum!
momentum
laplacian!
laplacian
pressuregradient!
pressuregradient
interpolate_u_p
interpolate_u_p!
interpolate_ω_p
interpolate_ω_p!
Dfield!
Dfield
Qfield!
Qfield
eig2field!
eig2field
kinetic_energy
kinetic_energy!
total_kinetic_energy
tensorbasis
divoftensor!
tensorbasis!
```
