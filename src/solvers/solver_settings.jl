"""
    SolverSettings(;
        pressure_solver = DirectPressureSolver(),
        p_add_solve = true,
    )

Solver settings.
"""
Base.@kwdef mutable struct SolverSettings{T}
    pressure_solver::PressureSolver = DirectPressureSolver() # PressureSolver
    p_add_solve::Bool = true                                 # Additional pressure solve to make it same order as velocity
end
