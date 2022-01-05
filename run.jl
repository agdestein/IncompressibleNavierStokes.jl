using GLMakie

# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
else
    using IncompressibleNavierStokes
end

## Load input parameters and constants
case_name = "LDC"
# case_name = "BFS"
# case_name = "TG"
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

## Post-process
plot_pressure(setup, p)
plot_vorticity(setup, V, setup.time.t_end)
plot_streamfunction(setup, V, setup.time.t_end)
