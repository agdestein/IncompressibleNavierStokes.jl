"""
    solve(problem, setup, V₀, p₀)

Solve `problem` with initial state `(V₀, p₀)`.
"""
function solve end

include("solve_steady_state.jl")
include("solve_unsteady.jl")
