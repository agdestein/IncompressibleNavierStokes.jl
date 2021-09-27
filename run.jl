using IncompressibleNavierStokes
using Plots

# Load input parameters and constants
case_name = "LDC"
include("case_files/$case_name.jl")
setup = eval(:($(Symbol(case_name))()))

V, p, setup, totaltime = main(setup)

