function pressure_additional_solve!(V, p, t, setup)
    @unpack NV = setup.grid

    cache = MomentumCache(setup)
    F = zeros(NV)

    pressure_additional_solve!(V, p, t, setup, cache, F)
end

"""
Additional pressure solve.
make the pressure compatible with the velocity field. this should
also result in same order pressure as velocity
"""
function pressure_additional_solve!(V, p, t, setup, cache, F)
    # Note: time derivative of BC in ydM
    @unpack M, ydM = setup.discretization
    @unpack Ω⁻¹ = setup.grid

    # Get updated BC for ydM
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    # Momentum already contains G*p with the current p, we therefore effectively solve for the pressure difference
    momentum!(F, nothing, V, V, p, t, setup, cache)
    f = M * (Ω⁻¹ .* F) + ydM
    Δp = pressure_poisson(f, t, setup)

    p .+= Δp
end
