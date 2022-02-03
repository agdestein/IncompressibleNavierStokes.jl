"""
    pressure_additional_solve!(V, p, t, setup)

Convenience function for allocating momentum cache, `F`, `Δp`, and `f` before doing
additional pressure solve.
"""
function pressure_additional_solve!(V, p, t, setup)
    (; NV, Np) = setup.grid

    momentum_cache = MomentumCache(setup)
    F = zeros(NV)
    Δp = zeros(Np)
    f = zeros(Np)

    pressure_additional_solve!(V, p, t, setup, momentum_cache, F, f, Δp)
end

"""
    pressure_additional_solve!(V, p, t, setup, momentum_cache, F, f)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure_additional_solve!(V, p, t, setup, momentum_cache, F, f, Δp)
    (; grid, operators, pressure_solver) = setup
    (; Ω⁻¹) = grid
    (; M) = operators

    # Get updated BC for ydM (time derivative of BC in ydM)
    # FIXME: `set_bc_vectors` are called to often (also inside `momentum!`)
    setup.bc.bc_unsteady && set_bc_vectors!(setup, t)
    (; ydM) = operators

    # Momentum already contains G*p with the current p, we therefore effectively solve for
    # the pressure difference
    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)

    # f = M * (Ω⁻¹ .* F) + ydM
    @. F = Ω⁻¹ .* F
    mul!(f, M, F)
    @. f = f + ydM

    pressure_poisson!(pressure_solver, Δp, f)

    p .+= Δp
end
