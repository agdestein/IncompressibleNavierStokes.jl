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
    # note: time derivative of BC in ydM
    @unpack M, ydM = setup.discretization
    @unpack Om_inv = setup.grid

    # get updated BC for ydM
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    # momentum already contains G*p with the current p, we therefore effectively solve for the pressure difference
    momentum!(F, nothing, V, V, p, t, setup, cache)
    f = M * (Om_inv .* F) + ydM
    Δp = pressure_poisson(f, t, setup)

    p .+= Δp
end
