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
    dotemp = !isnothing(temperature)
    nstage = length(b)

    # Update current solution
    tstart = t
    copyto!(statestart.u, state.u)
    dotemp && copyto!(statestart.temp, state.temp)

    for i = 1:nstage
        # Compute force at current stage i
        force!(k[i], state, t, params, setup, force_cache)

        # Apply stage forces
        copyto!(state.u, statestart.u)
        dotemp && copyto!(state.temp, statestart.temp)
        for j = 1:i
            @. state.u += Δt * A[i, j] * k[j].u
            dotemp && @. state.temp += Δt * A[i, j] * k[j].temp
        end

        # Intermediate time step
        t = tstart + c[i] * Δt

        # Project stage u directly
        # Make velocity divergence free at time t
        apply_bc_u!(state.u, t, setup)
        project!(state.u, setup; psolver, p)

        # Fill boundary values at new time
        apply_bc_u!(state.u, t, setup)
        dotemp && apply_bc_temp!(state.temp, t, setup)
    end

    create_stepper(method; setup, psolver, state, t, n = n + 1)
end

function timestep(method::ExplicitRungeKuttaMethod, force, stepper, Δt; params = nothing)
    (; setup, psolver, state, t, n) = stepper
    (; A, b, c) = method
    (; temperature) = setup
    dotemp = !isnothing(temperature)
    nstage = length(b)

    # Update current solution (does not depend on previous step size)
    tstart = t
    statestart = deepcopy(state)
    k = ()

    for i = 1:nstage
        # Compute force at current stage i
        f = force(state, t, params, setup)

        # Store right-hand side of stage i
        k = (k..., f)

        # Apply stage forces
        u = statestart.u
        dotemp && (temp = statestart.temp)
        for j = 1:i
            u = @. u + Δt * A[i, j] * k[j].u
            dotemp && (temp = @. temp + Δt * A[i, j] * k[j].temp)
        end

        # New time step
        t = tstart + c[i] * Δt

        # Project stage u directly
        # Make velocity divergence free at time t
        u = apply_bc_u(u, t, setup)
        u = project(u, setup; psolver)

        # Fill boundary values at new time
        u = apply_bc_u(u, t, setup)
        dotemp && (temp = apply_bc_temp(temp, t, setup))

        state = dotemp ? (; u, temp) : (; u)
    end

    create_stepper(method; setup, psolver, state, t, n = n + 1)
end
