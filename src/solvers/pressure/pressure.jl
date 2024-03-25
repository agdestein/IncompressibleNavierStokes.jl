"""
    pressure!(p, u, t, setup; psolver, F, div)

Compute pressure from velocity field. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure!(p, u, t, setup; psolver, F, div)
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    D = dimension()
    momentum!(F, u, t, setup)
    apply_bc_u!(F, t, setup; dudt = true)
    divergence!(div, F, setup)
    @. div *= Ω
    poisson!(psolver, p, div)
    apply_bc_p!(p, t, setup)
    p
end

"""
    pressure(u, t, setup; psolver)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure(u, t, setup; psolver)
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    D = dimension()
    F = momentum(u, t, setup)
    F = apply_bc_u(F, t, setup; dudt = true)
    div = divergence(F, setup)
    div = @. div * Ω
    p = poisson(psolver, div)
    p = apply_bc_p(p, t, setup)
    p
end
