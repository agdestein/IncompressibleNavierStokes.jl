abstract type Problem end

"""
    SteadyStateProblem(setup, V₀, p₀)

Steady state problem. The solution `V` and `p` should be such that ``\\frac{\\partial
\\mathbf{V}}{\\partial t} = \\mathbf{0}``.
"""
struct SteadyStateProblem{T,N} <: Problem
    setup::Setup{T,N}
    V₀::Vector{T}
    p₀::Vector{T}
end

"""
    UnsteadyProblem(setup, V₀, p₀, (t_start, t_stop))

Unsteady problem with initial conditions `V₀`, `p₀` to be solved from `t_start` to `t_stop`.
"""
struct UnsteadyProblem{T,N} <: Problem
    setup::Setup{T,N}
    V₀::Vector{T}
    p₀::Vector{T}
    tlims::Tuple{T,T}
end
