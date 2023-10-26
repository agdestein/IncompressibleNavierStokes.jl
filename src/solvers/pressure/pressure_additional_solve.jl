"""
    pressure_additional_solve!(pressure_solver, u, p, t, setup, F, f, M)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure_additional_solve!(pressure_solver, u, p, t, setup, F, G, M)
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    D = dimension()

    momentum!(F, u, t, setup)

    apply_bc_u!(F, t, setup; dudt = true)

    apply_bc_p!(p, t, setup)
    pressuregradient!(G, p, setup)
    for α = 1:D
        F[α] .-= G[α]
        # F[α][Iu[α]] .-= G[α][Iu[α]]
    end
    divergence!(M, F, setup)
    @. M *= Ω

    pressure_poisson!(pressure_solver, p, M)
    # dp = pressure_poisson(pressure_solver, M)
    # p .+= dp
    apply_bc_p!(p, t, setup)
    p
end

"""
    pressure_additional_solve(pressure_solver, u, t, setup)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure_additional_solve(pressure_solver, u, t, setup)
    D = setup.grid.dimension()
    p = KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N)
    F = ntuple(α -> KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N), D)
    G = ntuple(α -> KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N), D)
    M = KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N)
    pressure_additional_solve!(pressure_solver, u, p, t, setup, F, G, M)
end
