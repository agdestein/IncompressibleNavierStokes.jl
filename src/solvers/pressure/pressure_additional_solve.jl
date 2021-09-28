"""
Additional pressure solve.
make the pressure compatible with the velocity field. this should
also result in same order pressure as velocity
"""
function pressure_additional_solve(V, p, t, setup)
    # note: time derivative of BC in ydM
    @unpack M, ydM = setup.discretization
    @unpack Om_inv = setup.grid

    # get updated BC for ydM
    if setup.bc.bc_unsteady
        setup = set_bc_vectors(t, setup)
    end

    # F already contains G*p with the current p, we therefore effectively solve for the pressure difference
    _, R, _ = momentum(V, V, p, t, setup)
    f = M * (Om_inv .* R) + ydM
    Δp = pressure_poisson(f, t, setup)

    p .+= Δp
end
