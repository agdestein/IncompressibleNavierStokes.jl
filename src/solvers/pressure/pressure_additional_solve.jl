"""
    pressure_additional_solve(pressure_solver, V, p, t, setup)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure_additional_solve(pressure_solver, V, p, t, setup)
    (; grid, operators) = setup
    (; Ω⁻¹) = grid
    (; M) = operators

    # Get updated BC for ydM (time derivative of BC in ydM)
    # FIXME: `set_bc_vectors` are called to often (also inside `momentum!`)
    setup.bc.bc_unsteady && set_bc_vectors!(setup, t)
    (; ydM) = operators

    # Momentum already contains G*p with the current p, we therefore effectively solve for
    # the pressure difference
    F, = momentum(V, V, p, t, setup)
    f = M * (Ω⁻¹ .* F) + ydM

    Δp = pressure_poisson(pressure_solver, f)
    p .+ Δp
end

"""
    pressure_additional_solve!(pressure_solver, V, p, t, setup, momentum_cache, F, f)

Do additional pressure solve. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure_additional_solve!(pressure_solver, V, p, t, setup, momentum_cache, F, f, Δp)
    (; grid, operators) = setup
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
