using IncompressibleNavierStokes
using FFTW
using WGLMakie
using Random

T = Float64
n = 16
setup = Setup(;
    x = (
        range(0 |> T, 7 |> T, n + 1),
        range(0 |> T, 7 |> T, n + 1),
        range(-3 |> T, 2 |> T, 2n + 1),
    ),
    boundary_conditions = (
        (DirichletBC((0.0, 1.0)), DirichletBC()),
        # (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
);
(; Ip) = setup.grid

psolver = IncompressibleNavierStokes.psolver_transform(setup)

u = randn!(vectorfield(setup))
IncompressibleNavierStokes.apply_bc_u!(u, T(0), setup)
# u[2:end-1, :, 1] .-= sum(u[:, :, 1]; dims=1) / n
# u[:, 2:end-1, 2] .-= sum(u[:, :, 2]; dims=2) / n
# IncompressibleNavierStokes.apply_bc_u!(u, T(0), setup)

up = project(u, setup; psolver)
IncompressibleNavierStokes.apply_bc_u!(up, T(0), setup)

divergence(u, setup)
divergence(up, setup)
divergence(up, setup) |> extrema

psolver = IncompressibleNavierStokes.psolver_dct(setup)
p = scalarfield(setup)
randn!(view(p, Ip))
IncompressibleNavierStokes.apply_bc_p!(p, T(0), setup)
ps = laplacian(p, setup)

psolver(ps)

ps[Ip]
p[Ip]

psolver.ahat.contents[1]

T = Float64
n = 16
ax = range(0 |> T, 1 |> T, n + 1)
setup = Setup(;
    x = (ax, ax),
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
);

function lap(u, h)
    n = length(u)
    l = zero(u)
    for i = 2:(n-1)
        l[i] = (u[i-1] - 2 * u[i] + u[i+1]) / h^2
    end
    l[1] = (u[2] - u[1]) / h^2
    l[n] = -(u[n] - u[n-1]) / h^2
    l
end

n = 100
u = randn(n)
u[1] = 0
u[end] = 0
# u .-= sum(u) / n
h = L / n
uhat = dct(u)
k = 0:(n-1)
λ = @. 2 * (cospi(k / n) - 1) / h^2
phat = uhat ./ λ
phat[1] = 0
p = idct(phat)
lap(p, h) - u
