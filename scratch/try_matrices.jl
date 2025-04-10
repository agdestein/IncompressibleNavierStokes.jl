if false
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CairoMakie
using Random

x = tanh_grid(0.0, 5.0, 101), cosine_grid(0, 1, 61)
setup = Setup(; x, Re = 1e3)

m = IncompressibleNavierStokes.apply_bc_p_mat(PeriodicBC(), setup, 1, false)jG
m |> collect
p = scalarfield(setup)

m * p

randn!(p)

p1 = apply_bc_p(PeriodicBC(), setup, p, 1, false)

p = scalarfield(setup)
p[(end-10):(end-1), :] .= 1
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

ax = range(0, 1, 101)
setup = Setup(;
    x,
    Re = 1e3,
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
)

L1 = IncompressibleNavierStokes.laplacian_mat(setup)
L2 = IncompressibleNavierStokes.poisson_mat(setup)

L1 - L2 |> maximum
