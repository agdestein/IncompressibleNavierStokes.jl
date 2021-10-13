using IncompressibleNavierStokes
using GLMakie

## Load input parameters and constants
# Case_name = "LDC"
# case_name = "BFS"
case_name = "TG"
include("case_files/$case_name.jl")
setup = eval(:($(Symbol(case_name))()))

##
V, p, setup, totaltime, solution = main(setup);
