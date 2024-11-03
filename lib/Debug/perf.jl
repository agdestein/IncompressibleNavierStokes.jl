using IncompressibleNavierStokes
using CUDA
using BenchmarkTools
using Cthulhu
using KernelAbstractions

T = Float32
n = 512
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
