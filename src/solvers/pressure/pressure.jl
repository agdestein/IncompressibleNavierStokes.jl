"""
    pressure!(solver, u, p, t, setup, F, f, M)

Compute pressure from velocity field. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure!(solver, u, p, t, setup, F, G, M)
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    D = dimension()
    momentum!(F, u, t, setup)
    apply_bc_u!(F, t, setup; dudt = true)
    divergence!(M, F, setup)
    @. M *= Ω
    poisson!(solver, p, M)
    apply_bc_p!(p, t, setup)
    p
end

"""
    pressure(solver, u, t, setup)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure(solver, u, t, setup)
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    D = dimension()
    F = momentum(u, t, setup)
    F = apply_bc_u(F, t, setup; dudt = true)
    M = divergence(F, setup)
    M = @. M * Ω
    p = poisson(solver, M)
    p = apply_bc_p(p, t, setup)
    p
end
