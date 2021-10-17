using IncompressibleNavierStokes
using GLMakie

## Load input parameters and constants
# case_name = "LDC"
case_name = "BFS"
# case_name = "TG"
include("case_files/$case_name.jl")
setup = eval(:($(Symbol(case_name))()));

##
@time V, p, totaltime = main(setup);

##
@profview main(setup)

##

# Turbulence constants
if setup.case.visc == "keps"
    constants_ke!(setup)
end

# Construct mesh
create_mesh!(setup);

# Boundary conditions
create_boundary_conditions!(setup);

# Construct operators (matrices) which are time-independent
build_operators!(setup);

# Initialization of solution vectors
V₀, p₀, t₀ = create_initial_conditions(setup);

# Input checking
check_input!(setup, V₀, p₀, t₀)

# Solve problem
problem = setup.case.problem
@time V, p = solve(problem, setup, V₀, p₀);

# Measure total time
totaltime = Base.time() - starttime

## Post-processing
postprocess(setup, V, p, setup.time.t_end);
plot_pressure(setup, p);
plot_vorticity(setup, V, setup.time.t_end);
plot_streamfunction(setup, V, setup.time.t_end);
