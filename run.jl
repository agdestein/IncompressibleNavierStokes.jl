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
