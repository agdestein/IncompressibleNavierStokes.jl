using IncompressibleNavierStokes
using DispatchDoctor
using AllocCheck
using Cthulhu
using KernelAbstractions
using JET

ax = range(0, 1, 17)

x = (ax, ax)
boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x))
backend = IncompressibleNavierStokes.CPU()

function g()
    x = 1
    h() = (x = 2.0)
    h()
    return x
end

setup = Setup(; x = (ax, ax), Re = 1e3);
psolver = default_psolver(setup)

u = random_field(setup)

diffusion(u, setup)

@descend diffusion(u, setup)
@report_opt diffusion(u, setup)
