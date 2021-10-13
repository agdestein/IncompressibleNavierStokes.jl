"""
    reate_boundary_conditions!(setup)

Create discrete boundary condtions.
"""
function create_boundary_conditions!(setup)
    # Get BC type
    bc_type = setup.bc.bc_type()
    for (key, value) ∈ zip(keys(bc_type), bc_type)
        setfield!(setup.bc, key, value)
    end
    setup.bc.u.left ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-left")
    setup.bc.u.right ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-right")
    setup.bc.u.low ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-low")
    setup.bc.u.up ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-up")
    setup.bc.v.left ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-left")
    setup.bc.v.right ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-right")
    setup.bc.v.low ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-low")
    setup.bc.v.up ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-up")

    # Values set below can be either Dirichlet or Neumann value,
    # Depending on BC set above. in case of Neumann (symmetry, pressure)
    # One uses normally zero gradient

    # Values should either be scalars or vectors
    # ALL VALUES (u, v, p, k, e) are defined at x, y locations,
    # I.e. the corners of pressure volumes, so they cover the entire domain
    # Including corners

    ## Pressure
    # Pressure BC is only used when at the corresponding boundary
    # :pressure is specified
    p_inf = 0
    pLe = p_inf
    pRi = p_inf
    pLo = p_inf
    pUp = p_inf

    setup.bc.pLe = pLe
    setup.bc.pRi = pRi
    setup.bc.pLo = pLo
    setup.bc.pUp = pUp

    ## K-eps values
    if setup.case.visc == "keps"
        kLo = 0
        kUp = 0
        kLe = 0
        kRi = 0

        eLo = 0
        eUp = 0
        eLe = 0
        eRi = 0

        setup.bc.kLe = kLe
        setup.bc.kRi = kRi
        setup.bc.kLo = kLo
        setup.bc.kUp = kUp

        setup.bc.eLe = eLe
        setup.bc.eRi = eRi
        setup.bc.eLo = eLo
        setup.bc.eUp = eUp
    end

    setup
end
