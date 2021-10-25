"""
    reate_boundary_conditions!(setup)

Create discrete boundary condtions.

Values should either be scalars or vectors. All values `(u, v, p, k, e)` are defined at x, y locations,
i.e. the corners of pressure volumes, so they cover the entire domain, including corners.
"""
function create_boundary_conditions!(setup)
    @unpack model = setup

    # Get BC type
    bc_type = setup.bc.bc_type()
    for (key, value) ∈ zip(keys(bc_type), bc_type)
        setfield!(setup.bc, key, value)
    end
    setup.bc.u.x[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-left")
    setup.bc.u.x[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for u-right")
    setup.bc.u.y[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-low")
    setup.bc.u.y[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for u-up")
    setup.bc.v.x[1] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-left")
    setup.bc.v.x[2] ∈ [:dirichlet, :periodic, :symmetric] || error("Wrong BC for v-right")
    setup.bc.v.y[1] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-low")
    setup.bc.v.y[2] ∈ [:dirichlet, :periodic, :pressure] || error("Wrong BC for v-up")

    ## Pressure
    # Pressure BC is only used when at the corresponding boundary `:pressure` is specified
    p_inf = 0
    pLe = p_inf
    pRi = p_inf
    pLo = p_inf
    pUp = p_inf

    @pack! setup.bc = pLe, pRi, pLo, pUp

    ## K-eps values
    if model isa KEpsilonModel
        kLo = 0
        kUp = 0
        kLe = 0
        kRi = 0

        eLo = 0
        eUp = 0
        eLe = 0
        eRi = 0

        @pack! setup.bc = kLe, kRi, kLo, kUp
        @pack! setup.bc = eLe, eRi, eLo, eUp
    end

    setup
end
