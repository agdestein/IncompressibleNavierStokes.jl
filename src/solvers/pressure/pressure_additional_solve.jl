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
    # Note: time derivative of BC in ydM
    (; pressure_solver) = setup.solver_settings
    (; M, ydM) = setup.operators
    (; Ω⁻¹) = setup.grid

    # Get updated BC for ydM
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    # Momentum already contains G*p with the current p, we therefore effectively solve for
    # the pressure difference
    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)

    # f = M * (Ω⁻¹ .* F) + ydM
    @. F = Ω⁻¹ .* F
    mul!(f, M, F)
    @. f = f + ydM

    pressure_poisson!(pressure_solver, Δp, f, t, setup)

    p .+= Δp
end
