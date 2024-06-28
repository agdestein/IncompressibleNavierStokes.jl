using IncompressibleNavierStokes
using GLMakie
using SymmetryClosure
using LinearAlgebra

T = Float64
ArrayType = Array
n = 128
setup =
    Setup(LinRange(T(0), T(1), n + 1), LinRange(T(0), T(1), n + 1); Re = T(2000), ArrayType);
psolver = psolver_spectral(setup);

ustart = random_field(setup, T(0); psolver);

state, _ = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(0.5)),
    # Δt = T(1e-3),
    # Δt = T(2e-4),
    processors = (
        rtp = realtimeplotter(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

ustart_rot = rot2(ustart, setup);

state_rot, _ = solve_unsteady(;
    setup,
    ustart = ustart_rot,
    tlims = (T(0), T(0.5)),
    # Δt = T(1e-3),
    # Δt = T(2e-4),
    processors = (
        rtp = realtimeplotter(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);

F = IncompressibleNavierStokes.momentum(ustart, nothing, T(0), setup)
FR = IncompressibleNavierStokes.momentum(ustart_rot, nothing, T(0), setup)

F = IncompressibleNavierStokes.diffusion(ustart, setup)
FR = IncompressibleNavierStokes.diffusion(ustart_rot, setup)

F = IncompressibleNavierStokes.convection(ustart, setup)
FR = IncompressibleNavierStokes.convection(ustart_rot, setup)

RF = rot2(F, setup)

i = 1
# a = RF[i]
# b = FR[i]
a = RF[i][setup.grid.Iu[i]];
b = FR[i][setup.grid.Iu[i]];
norm(a - b) / norm(b)

a = RF[i][setup.grid.Iu[i]]
b = FR[i][setup.grid.Iu[i]]

heatmap(a)
heatmap(b)

i = 1
a = state_rot.u[i][setup.grid.Iu[i]]
b = rot2(state.u, setup)[i][setup.grid.Iu[i]]

hist(a[:])
hist!((b.-a)[:])

norm(a - b) / norm(b)

st(u) = (; u, t = T(0), temp = nothing)

GLMakie.closeall()
s1 = GLMakie.Screen()
s2 = GLMakie.Screen()

display(s1, fieldplot(st(ustart); setup, fieldname = 1))
display(s2, fieldplot(st(ustart_rot); setup, fieldname = 2))

display(s1, fieldplot(
    st(rot2(state.u, setup));
    setup,
    # fieldname = 2,
))
display(s2, fieldplot(
    state_rot;
    setup,
    # fieldname = 2,
))

display(GLMakie.Screen(), fieldplot(st(rot2(state.u)); setup))
display(GLMakie.Screen(), fieldplot(state_rot; setup))

x = [1 2 3; 0 0 9; 0 0 1]
circshift(x, (1, 0))
rot2(x)
