create_stepper(::LMWray3; setup, psolver, state, t, n = 0) = (; setup, psolver, state, t, n)

function timestep!(
    method::LMWray3,
    force!,
    stepper,
    Δt;
    params = nothing,
    ode_cache,
    force_cache,
)
    (; setup, psolver, state, t, n) = stepper
    (; temperature) = setup
    (; statestart, p, k) = ode_cache
    T = eltype(state.u)

    dotemp = !isnothing(temperature)

    # Boundary conditions and pressure projection
    function correct!(state, t, setup)
        # Project stage u directly
        # Make velocity divergence free at time t
        apply_bc_u!(state.u, t, setup)
        project!(state.u, setup; psolver, p)
        state
    end

    # Copy state x to y
    function state_copyto!(y, x)
        copyto!(y.u, x.u)
        dotemp && copyto!(y.temp, x.temp)
        y
    end

    # Compute y = a * x + y for states x, y
    function state_axpy!(a, x, y)
        axpy!(a, x.u, y.u)
        dotemp && axpy!(a, x.temp, y.temp)
    end

    # Update current solution
    tstart = t
    state_copyto!(statestart, state)

    # Low-storage Butcher tableau:
    # c1 | 0             ⋯   0
    # c2 | a1  0         ⋯   0
    # c3 | b1 a2  0      ⋯   0
    # c4 | b1 b2 a3  0   ⋯   0
    #  ⋮ | ⋮   ⋮  ⋮  ⋱   ⋱   ⋮
    # cn | b1 b2 b3  ⋯ an-1  0
    # ---+--------------------
    #    | b1 b2 b3  ⋯ bn-1 an
    #
    # Note the definition of (ai)i.
    # They are shifted to simplify the for-loop.
    # TODO: Make generic by passing a, b, c as inputs
    a = T(8 / 15), T(5 / 12), T(3 / 4)
    b = T(1 / 4), T(0)
    c = T(0), T(8 / 15), T(2 / 3)
    nstage = length(a)

    for i = 1:nstage
        force!(k, state, t, params, setup, force_cache)

        # Compute x = correct(xstart + Δt * a[i] * dx)
        t = tstart + c[i] * Δt
        state_copyto!(state, statestart)
        state_axpy!(a[i] * Δt, k, state)
        correct!(state, t, setup)

        # Compute statestart = statestart + Δt * b[i] * k
        # Skip for last iter
        i == nstage || state_axpy!(b[i] * Δt, k, statestart)

        # Fill boundary values at new time
        apply_bc_u!(state.u, t, setup)
        dotemp && apply_bc_temp!(state.temp, t, setup)
    end

    # Full time step
    t = tstart + Δt

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    apply_bc_u!(state.u, t, setup)
    dotemp && apply_bc_temp!(state.temp, t, setup)

    create_stepper(method; setup, psolver, state, t, n = n + 1)
end

function timestep(method::LMWray3, force, stepper, Δt; params = nothing)
    (; setup, psolver, state, t, n) = stepper
    (; temperature) = setup
    (; u) = state
    T = eltype(u)

    dotemp = !isnothing(temperature)

    # Update current state
    tstart = t
    ustart = state.u
    dotemp && (tempstart = state.temp)

    # Low-storage Butcher tableau:
    # c1 | 0             ⋯   0
    # c2 | a1  0         ⋯   0
    # c3 | b1 a2  0      ⋯   0
    # c4 | b1 b2 a3  0   ⋯   0
    #  ⋮ | ⋮   ⋮  ⋮  ⋱   ⋱   ⋮
    # cn | b1 b2 b3  ⋯ an-1  0
    # ---+--------------------
    #    | b1 b2 b3  ⋯ bn-1 an
    #
    # Note the definition of (ai)i.
    # They are shifted to simplify the for-loop.
    # TODO: Make generic by passing a, b, c as inputs
    a = T(8 / 15), T(5 / 12), T(3 / 4)
    b = T(1 / 4), T(0)
    c = T(0), T(8 / 15), T(2 / 3)
    nstage = length(a)

    for i = 1:nstage
        k = force(state, t, params, setup)

        # Compute state at current stage
        t = tstart + c[i] * Δt
        u = @. ustart + Δt * a[i] * k.u
        u = apply_bc_u(u, t, setup)
        u = project(u, setup; psolver)
        if dotemp
            temp = @. tempstart + Δt * a[i] * k.temp
        end

        # Advance start state (skip for last iter)
        if i < nstage
            ustart = @. ustart + Δt * b[i] * k.u
            if dotemp
                tempstart = @. tempstart + Δt * b[i] * k.temp
            end
        end

        # Fill boundary values at new time
        u = apply_bc_u(u, t, setup)
        if dotemp
            temp = apply_bc_temp(temp, t, setup)
        end

        state = dotemp ? (; u, temp) : (; u)
    end

    # Full time step
    t = tstart + Δt

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term

    create_stepper(method; setup, psolver, state, t, n = n + 1)
end
