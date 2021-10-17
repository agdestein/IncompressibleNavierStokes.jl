"""
    is_steady(problem)

Returns `true` if problem is unsteady.
"""
function is_steady end

# By deafault, assume problem to be unsteady
is_steady(::Problem) = false
is_steady(::SteadyStateProblem) = true
