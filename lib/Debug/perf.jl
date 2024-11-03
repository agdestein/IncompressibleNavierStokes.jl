using IncompressibleNavierStokes
using CUDA
using BenchmarkTools
using Cthulhu
using KernelAbstractions

T = Float32
n = 128
ax = range(T(0), T(1), n + 1)
setup = Setup(; x = (ax, ax, ax), Re = T(1e3), backend = CUDABackend())
u = random_field(setup)

p = scalarfield(setup)
let
    fill!(p, 0)
    @benchmark IncompressibleNavierStokes.divergence!($p, $u, $setup)
end

f = vectorfield(setup)
let
    fill!.(f, 0)
    @benchmark IncompressibleNavierStokes.diffusion!($f, $u, $setup)
end

setup.workgroupsize

IncompressibleNavierStokes.convection!(f, u, setup)

f[1]

@descend IncompressibleNavierStokes.diffusion!(f, u, setup)

f = vectorfield(setup)
g = vectorfield(setup)
let
    fill!.(f, 0)
    fill!.(g, 0)
    @benchmark IncompressibleNavierStokes.convection_adjoint!($f, $g, $u, $setup)
end

f = vectorfield(setup);
let
    fill!.(f, 0)
    @benchmark IncompressibleNavierStokes.convectiondiffusion!($f, $u, $setup)
end

uu = stack(u);
ff = copy(uu);
let
    fill!(ff, 0)
    @benchmark IncompressibleNavierStokes.arrayconvectiondiffusion!($ff, $uu, $setup)
end

f = vectorfield(setup)
p = scalarfield(setup)
let
    fill!.(f, 0)
    @benchmark IncompressibleNavierStokes.pressuregradient!($f, $p, $setup)
end
