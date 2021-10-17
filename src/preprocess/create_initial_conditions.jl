"""
    V, p = create_initial_conditions(setup)

Create initial vectors.
"""
function create_initial_conditions(setup)
    @unpack problem = setup.case
    @unpack xu, yu, xv, yv, xpp, ypp = setup.grid
    @unpack Ω⁻¹, NV = setup.grid
    @unpack G, M, yM = setup.discretization

    t = setup.time.t_start

    # Boundary conditions
    set_bc_vectors!(setup, t)

    # Construct body force or immersed boundary method
    # The body force is called in the residual routines e.g. momentum.jl
    # Steady force can be precomputed once:
    if setup.force.isforce
        Fx, Fy, _ = bodyforce(V, t, setup)
        F = [Fx; Fy]
    else
        F = zeros(NV)
        @pack! setup.force = F
    end

    # Allocate velocity and pressure
    u = zero(xu)
    v = zero(xv)
    p = zero(xpp)

    # Initial velocities
    u .= setup.case.initial_velocity_u.(xu, yu, [setup])
    v .= setup.case.initial_velocity_v.(xv, yv, [setup])
    V = [u[:]; v[:]]

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv, umom, vmom, k = compute_conservation(V, t, setup)

    if maxdiv > 1e-12 && !is_steady(problem)
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv. Performing additional projection."

        # Make velocity field divergence free
        f = M * V + yM
        Δp = pressure_poisson(pressure_solver, f, t, setup)
        V .-= Ω⁻¹ .* (G * Δp)

        # Repeat conservation with updated velocity field
        maxdiv, umom, vmom, k = compute_conservation(V, t, setup)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    p .= setup.case.initial_pressure.(xpp, ypp, [setup])
    p = p[:]
    if is_steady(problem)
        # For steady state computations, the initial guess is the provided initial condition
    else
        if setup.solver_settings.p_initial
            # Calculate initial pressure from a Poisson equation
            pressure_additional_solve!(V, p, t, setup)
        else
            # Use provided initial condition (not recommended)
        end
    end

    V, p, t
end
