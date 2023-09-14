"""
    pressure_additional_solve!(pressure_solver, u, p, t, setup, F, f, M)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure_additional_solve!(pressure_solver, u, p, t, setup, F, M)
    (; grid) = setup
    (; Ip) = grid

    momentum!(F, u, t, setup)
    apply_bc_u!(F, t, setup; dudt = true)
    divergence!(M, F, setup)

    Min = view(M, Ip)
    pin = view(p, Ip)
    pressure_poisson!(pressure_solver, pin, Min)
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
    M = KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N)
    pressure_additional_solve!(pressure_solver, u, p, t, setup, F, M)
end
