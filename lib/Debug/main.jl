using IncompressibleNavierStokes
using DispatchDoctor
using AllocCheck
using Cthulhu
using KernelAbstractions
using JET

ax = range(0, 1, 17)

x = (ax, ax)
boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x))
backend = CPU()

@descend IncompressibleNavierStokes.Grid(; x, boundary_conditions, backend)

@descend Setup(; x = (ax, ax), Re = 1e3);
@report_opt Setup(; x = (ax, ax), Re = 1e3)

@descend IncompressibleNavierStokes.Dimension(3)

@report_opt IncompressibleNavierStokes.Dimension(3)

@report_opt IncompressibleNavierStokes.Grid(; x, boundary_conditions, backend)

f(x) =
    map(x) do x
        Δ = diff(x)
        # Δ[Δ.==0] .= eps(T)
        # Δ = min.(Δ, eps(T))
        Δ
    end

x = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
@descend f(x)

y = [1.0, 2.0]
@descend diff(y)

function g()
    x = 1
    h() = (x = 2.0)
    h()
    return x
end

g()


setup = Setup(; x = (ax, ax), Re = 1e3);
psolver = default_psolver(setup)

@descend default_psolver(setup)


u = vectorfield(setup)
p = scalarfield(setup)

u = random_field(setup)

diffusion(u, setup)
@descend diffusion(u, setup)
