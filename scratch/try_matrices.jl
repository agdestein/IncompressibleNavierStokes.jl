using IncompressibleNavierStokes
using CairoMakie
using Random

ax = range(0, 1, 101)
setup = Setup(; x = (ax, ax), Re = 1e3)


m = IncompressibleNavierStokes.apply_bc_p_mat(PeriodicBC(), setup, 1, false)
m |> collect
p = scalarfield(setup)

m * p

randn!(p)

p1 = apply_bc_p(PeriodicBC(), setup, p, 1, false)

p = scalarfield(setup)
p[end-10:end-1, :] .= 1
p |> heatmap

p1 = reshape(m * p[:], size(p))
p1 |> heatmap

B = IncompressibleNavierStokes.bc_p_mat(setup)
p = scalarfield(setup)
randn!(p)
p1 = reshape(B * p[:], size(p))

B = IncompressibleNavierStokes.bc_u_mat(setup)
u = vectorfield(setup)
randn!(u)
u1 = reshape(B * u[:], size(u))

B = IncompressibleNavierStokes.bc_u_mat(setup)
D = IncompressibleNavierStokes.diffusion_mat(setup)
u = vectorfield(setup)
randn!(u)

d1 = reshape(D * B * u[:], size(u)) / setup.Re
d2 = diffusion(apply_bc_u(u, 0, setup), setup)
