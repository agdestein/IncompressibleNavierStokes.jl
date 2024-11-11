create_stepper(::LMWray3; setup, psolver, u, temp, t, n = 0) =
    (; setup, psolver, u, temp, t, n)

function timestep!(method::LMWray3, stepper, Δt; θ = nothing, cache)
    (; setup, psolver, u, temp, t, n) = stepper
    (; closure_model, temperature) = setup
    (; ustart, ku, p, tempstart, ktemp, diff) = cache
    m = closure_model
    T = eltype(u)

    # We wrap the state in x = (; u, temp), and define some
    # functions that operate on x

    # Right-hand side function (without projection)
    function f!(dx, x, t, setup)
        # Velocity equation
        apply_bc_u!(x.u, t, setup)
        isnothing(x.temp) || apply_bc_temp!(x.temp, t, setup)
        momentum!(dx.u, x.u, x.temp, t, setup)

        # Add closure term
        isnothing(m) || (dx.u .+= m(x.u, θ))

        # Temperature equation
        if !isnothing(x.temp)
            fill!(dx.temp, 0)
            convection_diffusion_temp!(dx.temp, x.u, x.temp, setup)
            temperature.dodissipation && dissipation!(dx.temp, diff, x.u, setup)
        end

        dx
    end

    # Boundary conditions and pressure projection
    function correct!(x, t, setup)
        # Project stage u directly
        # Make velocity divergence free at time t
        apply_bc_u!(x.u, t, setup)
        project!(x.u, setup; psolver, p)
        x
    end

    # Copy state x to y
    function state_copyto!(y, x)
        copyto!(y.u, x.u)
        isnothing(temp) || copyto!(y.temp, x.temp)
        y
    end

    # Compute y = a * x + y for states x, y
    function state_axpy!(a, x, y)
        axpy!(a, x.u, y.u)
        isnothing(temp) || axpy!(a, x.temp, y.temp)
    end

    # States
    xstart = (; u = ustart, temp = tempstart)
    x = (; u, temp)
    dx = (; u = ku, temp = ktemp)

    # Update current solution
    tstart = t
    state_copyto!(xstart, x)

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
        t = tstart + c[i] * Δt
        f!(dx, x, t, setup)

        # Compute x = correct(xstart + Δt * a[i] * dx)
        state_copyto!(x, xstart)
        state_axpy!(a[i] * Δt, dx, x)
        correct!(x, t, setup)

        # Compute xstart = xstart + Δt * b[i] * dx
        # Skip for last iter
        i == nstage || state_axpy!(b[i] * Δt, dx, xstart)
    end

    # Full time step
    t = tstart + Δt

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    apply_bc_u!(x.u, t, setup)
    isnothing(x.temp) || apply_bc_temp!(x.temp, t, setup)

    create_stepper(method; setup, psolver, x.u, x.temp, t, n = n + 1)
end

function timestep(method::LMWray3, stepper, Δt; θ = nothing)
    (; setup, psolver, u, temp, t, n) = stepper
    (; closure_model, temperature) = setup
    m = closure_model
    T = eltype(u)

    # We wrap the state in x = (; u, temp), and define some
    # functions that operate on x

    # Right-hand side function (without projection)
    function f(u, temp, t, setup)
        u = apply_bc_u(u, t, setup)
        if isnothing(temp)
            dtemp = nothing
        else
            temp = apply_bc_temp(temp, t, setup)
            dtemp = convection_diffusion_temp(u, temp, setup)
            if temperature.dodissipation
                dtemp += dissipation(u, setup)
            end
        end
        du = momentum(u, temp, t, setup)

        # Add closure term
        isnothing(m) || (du += m(u, θ))

        du, dtemp
    end

    # Update current state
    tstart = t
    ustart = u
    tempstart = temp

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
        t = tstart + c[i] * Δt
        du, dtemp = f(u, temp, t, setup)

        # Compute state at current stage
        u = @. ustart + Δt * a[i] * du
        u = apply_bc_u(u, t, setup)
        u = project(u, setup; psolver)
        if !isnothing(temp)
            temp = @. tempstart + Δt * a[i] * dtemp
        end

        # Advance start state (skip for last iter)
        if i < nstage
            ustart = @. ustart + Δt * b[i] * du
            if !isnothing(temp)
                tempstart = @. tempstart + Δt * b[i] * dtemp
            end
        end
    end

    # Full time step
    t = tstart + Δt

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    u = apply_bc_u(u, t, setup)
    if !isnothing(temp)
        temp = apply_bc_temp(temp, t, setup)
    end

    create_stepper(method; setup, psolver, u, temp, t, n = n + 1)
end
