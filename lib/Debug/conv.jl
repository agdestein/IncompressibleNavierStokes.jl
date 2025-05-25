using IncompressibleNavierStokes
using CUDA
using Random
using BenchmarkTools

backend = CUDABackend()
T = Float64
ax = range(0 |> T, 1 |> T, 256)
setup = Setup(;
    x = (ax, ax, ax),
    boundary_conditions = (;
        u = (
            (PeriodicBC(), PeriodicBC()),
            (PeriodicBC(), PeriodicBC()),
            (PeriodicBC(), PeriodicBC()),
        ),
    ),
    backend,
);

u = randn!(vectorfield(setup));
f = randn!(vectorfield(setup));
p = randn!(scalarfield(setup));

@benchmark begin
    fill!(f, 0);
    # fill!(p, 0);
    IncompressibleNavierStokes.convection!(f, u, setup);
    # IncompressibleNavierStokes.convection2!(f, u, setup);
end

f[10, 10, 10:20, 1]

IncompressibleNavierStokes.left(CartesianIndex(10, 5), 1, 4)
