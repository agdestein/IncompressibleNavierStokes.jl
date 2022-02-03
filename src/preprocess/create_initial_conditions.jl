"""
    create_initial_conditions(
        setup,
        t;
        initial_velocity_u,
        initial_velocity_v,
        [initial_velocity_w,]
        initial_pressure = nothing,
    )

Create initial vectors at starting time `t`. If `p_initial` is a function instead of
`nothing`, calculate compatible IC for the pressure.
"""
function create_initial_conditions end

# 2D version
function create_initial_conditions(
    setup::Setup{T,2},
    t;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure = nothing,
) where {T}
    (; grid, pressure_solver) = setup
    (; xu, yu, xv, yv, xpp, ypp, Ω⁻¹) = grid

    # Boundary conditions
    set_bc_vectors!(setup, t)

    # Allocate velocity and pressure
    u = zero(xu)
    v = zero(xv)
    p = zero(xpp)

    # Initial velocities
    u .= initial_velocity_u.(xu, yu)
    v .= initial_velocity_v.(xv, yv)
    V = [u[:]; v[:]]

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv, umom, vmom, k = compute_conservation(V, t, setup)

    if maxdiv > 1e-12
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv.\n" *
              "Performing additional projection."

        # Make velocity field divergence free
        (; G, M, yM) = setup.operators
        f = M * V + yM
        Δp = pressure_poisson(pressure_solver, f, setup)
        V .-= Ω⁻¹ .* (G * Δp)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    isnothing(initial_pressure) || (p .= initial_pressure.(xpp, ypp))
    p = p[:]

    # For steady state computations, the initial guess is the provided initial condition
    isnothing(initial_pressure) || pressure_additional_solve!(V, p, t, setup)

    V, p
end

# 3D version
function create_initial_conditions(
    setup::Setup{T,3},
    t;
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure = nothing,
) where {T}
    (; grid, pressure_solver) = setup
    (; xu, yu, zu, xv, yv, zv, xw, yw, zw, xpp, ypp, zpp, Ω⁻¹) = grid

    # Boundary conditions
    set_bc_vectors!(setup, t)

    # Allocate velocity and pressure
    u = zero(xu)
    v = zero(xv)
    w = zero(xw)
    p = zero(xpp)

    # Initial velocities
    u .= initial_velocity_u.(xu, yu, zu)
    v .= initial_velocity_v.(xv, yv, zv)
    w .= initial_velocity_w.(xw, yw, zw)
    V = [u[:]; v[:]; w[:]]

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv, umom, vmom, wmom, k = compute_conservation(V, t, setup)

    if maxdiv > 1e-12
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv.\n" *
              "Performing additional projection."

        # Make velocity field divergence free
        (; G, M, yM) = setup.operators
        f = M * V + yM
        Δp = pressure_poisson(pressure_solver, f, setup)
        V .-= Ω⁻¹ .* (G * Δp)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    isnothing(initial_pressure) || (p .= initial_pressure.(xpp, ypp, zpp))
    p = p[:]

    # For steady state computations, the initial guess is the provided initial condition
    isnothing(initial_pressure) || pressure_additional_solve!(V, p, t, setup)

    V, p
end
