"""
    create_right_hand_side(setup, psolver)

Creates a function that computes the right-hand side of the Navier-Stokes equations for a given setup and pressure solver.

# Arguments
- `setup`: The simulation setup containing grid and boundary conditions.
- `psolver`: The pressure solver to be used.

# Returns
A function that computes the right-hand side of the Navier-Stokes equations.
"""
create_right_hand_side(setup, psolver) = function right_hand_side(u, params, t)
    # F = zeros(size(u))
    u = apply_bc_u(u, t, setup)
    f = navierstokes((; u), t; setup, params.viscosity)
    du = apply_bc_u(f.u, t, setup; dudt = true)
    du = project(du, setup; psolver)
end

"""
    right_hand_side!(dudt, u, params_ref, t)

Computes the right-hand side of the Navier-Stokes equations in-place.

# Arguments
- `dudt`: The array to store the computed right-hand side.
- `u`: The current velocity field.
- `params_ref`: A reference to the parameters containing the setup and pressure solver.
- `t`: The current time.

# Returns
Nothing. The result is stored in `dudt`.
"""
function right_hand_side!(dudt, u, params_ref, t)
    params = params_ref[]
    setup = params[1]
    psolver = params[2]
    viscosity = params[3]
    p = scalarfield(setup)
    # [!]*** be careful to not touch u in this function!
    temp_vector = copy(u)
    apply_bc_u!(temp_vector, t, setup)
    navierstokes!((; u = dudt), (; u = temp_vector), t; setup, viscosity, cache = nothing)
    apply_bc_u!(dudt, t, setup; dudt = true)
    project!(dudt, setup; psolver, p)
    return nothing
end
