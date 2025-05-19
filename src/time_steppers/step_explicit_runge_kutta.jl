create_stepper(::ExplicitRungeKuttaMethod; setup, psolver, state, t, n = 0) =
    (; setup, psolver, state, t, n)

function timestep!(
    method::ExplicitRungeKuttaMethod,
    force!,
    stepper,
    Δt;
    params = nothing,
    ode_cache,
    force_cache,
)
    (; setup, psolver, state, t, n) = stepper
    (; temperature) = setup
    (; A, b, c) = method
    (; statestart, k, p) = ode_cache
    nstage = length(b)

    # Update current solution
    tstart = t
    map(copyto!, statestart, state)

    for i = 1:nstage
        # Compute force at current stage i
        force!(k[i], state, t, params, setup, force_cache)

        # Apply stage forces
        map(copyto!, state, statestart)
        for j = 1:i
            for (u, k) in zip(state, k[j])
                @. u += Δt * A[i, j] * k
            end
        end

        # Intermediate time step
        t = tstart + c[i] * Δt

        # Project stage u directly
        # Make velocity divergence free at time t
        apply_bc_u!(state.u, t, setup)
        project!(state.u, setup; psolver, p)

        # Fill boundary values at new time
        for (key, field) in pairs(state)
            if key == :u
                apply_bc_u!(state.u, t, setup)
            elseif key == :temp
                apply_bc_temp!(field, t, setup)
            else
                # Fallback (empty scalar BC)
                # TODO: Rethink how to choose BC for different fields
                apply_bc_p!(field, t, setup)
            end
        end
    end

    create_stepper(method; setup, psolver, state, t, n = n + 1)
end

function timestep(method::ExplicitRungeKuttaMethod, force, stepper, Δt; params = nothing)
    (; setup, psolver, state, t, n) = stepper
    (; A, b, c) = method
    (; temperature) = setup
    nstage = length(b)

    # TODO: allow for different fields
    @assert all(f -> f in (:u, :temp), keys(state))
    dotemp = haskey(state, :temp)

    # Update current solution (does not depend on previous step size)
    tstart = t
    statestart = map(copy, state)
    k = ()

    for i = 1:nstage
        # Compute force at current stage i
        f = force(state, t, params, setup)

        # Store right-hand side of stage i
        k = (k..., f)

        # Apply stage forces
        state = statestart
        for j = 1:i
            state = map(state, k[j]) do field, k
                @. field + Δt * A[i, j] * k
            end
        end

        # New time step
        t = tstart + c[i] * Δt

        # Project stage u directly
        # Make velocity divergence free at time t
        u = apply_bc_u(u, t, setup)
        u = project(u, setup; psolver)

        # Fill boundary values at new time
        # TODO: automate for fields other than temp
        u = apply_bc_u(u, t, setup)
        dotemp && (temp = apply_bc_temp(state.temp, t, setup))

        state = dotemp ? (; u, temp) : (; u)
    end

    create_stepper(method; setup, psolver, state, t, n = n + 1)
end
