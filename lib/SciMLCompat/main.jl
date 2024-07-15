using SciMLCompat
using IncompressibleNavierStokes
using OrdinaryDiffEq
using GLMakie
# using CairoMakie

# Setup
Re = 2e3
n = 128
x = LinRange(0.0, 1.0, n + 1), LinRange(0.0, 1.0, n + 1);
setup = Setup(x...; Re);
ustart = random_field(setup, 0.0);
psolver = psolver_spectral(setup);

# SciML-compatible right hand side function
# Note: Requires `stack(u)` to create one array
f = create_right_hand_side(setup, psolver)
f(stack(ustart), nothing, 0.0)

# Solve the ODE using SciML
prob = ODEProblem(f, stack(ustart), (0.0, 1))
sol = solve(
    prob,
    Tsit5();
    # adaptive = false,
    dt = 1e-3,
)
sol.t

# Animate solution
let
    (; Iu) = setup.grid
    i = 1
    obs = Observable(sol.u[1][Iu[i], i])
    fig = heatmap(obs)
    fig |> display
    for u in sol.u
        obs[] = u[Iu[i], i]
        # fig |> display
        sleep(0.05)
    end
end
