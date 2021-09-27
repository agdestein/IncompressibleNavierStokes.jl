"""
    V, p = create_initial_conditions(setup)

Create initial vectors.
"""
function create_initial_conditions(setup)
    t = setup.time.t_start;

    # Grid points for u, v, and p
    @unpack xu, yu, xv, yv, xpp, ypp = setup.grid

    # constant velocity field
    u = setup.case.initial_velocity_u.(xu, yu, [setup])
    v = setup.case.initial_velocity_v.(xv, yv, [setup])

    # pressure: should in principle NOT be prescribed. Will be calculated if p_initial
    p = setup.case.initial_pressure.(xpp, ypp, [setup])

    # Stack all u components and v components
    V = [u[:]; v[:]];

    V, p, t
end
