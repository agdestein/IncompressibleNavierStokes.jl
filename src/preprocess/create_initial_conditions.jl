"""
    V, p = create_initial_conditions(setup)

Create initial vectors.
"""
function create_initial_conditions(setup)
    (; problem) = setup.case
    (; xu, yu, zu, xv, yv, zv, xw, yw, zw, xpp, ypp, zpp) = setup.grid
    (; Ω⁻¹) = setup.grid
    (; pressure_solver) = setup.solver_settings

    t = setup.time.t_start

    # Boundary conditions
    set_bc_vectors!(setup, t)

    # Allocate velocity and pressure
    u = zero(xu)
    v = zero(xv)
    w = zero(xw)
    p = zero(xpp)

    # Initial velocities
    u .= setup.case.initial_velocity_u.(xu, yu, zu)
    v .= setup.case.initial_velocity_v.(xv, yv, zv)
    w .= setup.case.initial_velocity_w.(xw, yw, zw)
    V = [u[:]; v[:]; w[:]]

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv, umom, vmom, wmom, k = compute_conservation(V, t, setup)

    if maxdiv > 1e-12 && !is_steady(problem)
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv. Performing additional projection."

        # Make velocity field divergence free
        (; G, M, yM) = setup.discretization
        f = M * V + yM
        Δp = pressure_poisson(pressure_solver, f, t, setup)
        V .-= Ω⁻¹ .* (G * Δp)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    p .= setup.case.initial_pressure.(xpp, ypp, zpp)
    p = p[:]

    # For steady state computations, the initial guess is the provided initial condition
    if !is_steady(problem) && setup.solver_settings.p_initial
        pressure_additional_solve!(V, p, t, setup)
    end

    V, p, t
end
