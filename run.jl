using IncompressibleNavierStokes
using GLMakie

## Load input parameters and constants
# case_name = "LDC"
# case_name = "BFS"
# case_name = "TG"
case_name = "TG3D"
include("case_files/$case_name.jl")
setup = eval(:($(Symbol(case_name))()));

## Prepare
build_operators!(setup);
V₀, p₀, t₀ = create_initial_conditions(setup);

## Solve problem
problem = setup.case.problem;
@time V, p = solve(problem, setup, V₀, p₀);

## Plot tracers
plot_tracers(setup);

##
@time V, p = solve(problem, setup, V, p);

##
@profview V, p = solve(problem, setup, V₀, p₀);

## Post-process
plot_pressure(setup, p)
plot_vorticity(setup, V, setup.time.t_end)
plot_streamfunction(setup, V, setup.time.t_end)
