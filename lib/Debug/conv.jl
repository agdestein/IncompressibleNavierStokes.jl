using IncompressibleNavierStokes
using CUDA
using Random
using BenchmarkTools

backend = CUDABackend()
T = Float64
ax = range(0 |> T, 1 |> T, 256)
setup = Setup(; x = (ax, ax, ax), backend);

u = randn!(vectorfield(setup));
f = vectorfield(setup);

IncompressibleNavierStokes.convection!(f, u, setup);
IncompressibleNavierStokes.convection2!(f, u, setup);

@benchmark begin
    fill!(f, 0)
    # IncompressibleNavierStokes.convection!(f, u, setup);
    IncompressibleNavierStokes.convection2!(f, u, setup);
    # IncompressibleNavierStokes.KernelAbstractions.synchronize(backend)
end
