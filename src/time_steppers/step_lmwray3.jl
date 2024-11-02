create_stepper(::LMWray3; setup, psolver, u, temp, t, n = 0) =
    (; setup, psolver, u, temp, t, n)

function timestep!(method::LMWray3, stepper, Δt; θ = nothing, cache)
    (; setup, psolver, u, temp, t, n) = stepper
    (; grid, closure_model, temperature) = setup
    (; dimension, Iu) = grid
    (; ustart, ku, p, tempstart, ktemp, diff) = cache
    D = dimension()
    m = closure_model
    T = eltype(u[1])

    # We wrap the state in x = (; u, temp), and define some
    # functions that operate on x

    # Right-hand side function (without projection)
    function f!(dx, x, t, setup)
        (; u, temp) = x
        apply_bc_u!(u, t, setup)
        isnothing(temp) || apply_bc_temp!(temp, t, setup)
        momentum!(dx.u, u, temp, t, setup)
        if !isnothing(temp)
            dx.temp .= 0
            convection_diffusion_temp!(dx.temp, u, temp, setup)
            temperature.dodissipation && dissipation!(dx.temp, diff, u, setup)
        end

        # Add closure term
        isnothing(m) || map((du, m) -> du .+= m, dx.u, m(u, θ))

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
        copyto!.(y.u, x.u)
        isnothing(temp) || copyto!(y.temp, x.temp)
        y
    end

    # Compute y = a * x + y for states x, y
    function state_axpy!(a, x, y)
        for α = 1:D
            @. y.u[α] += a * x.u[α]
        end
        if !isnothing(temp)
            @. y.temp += a * x.temp
        end
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
    #  --+--------------------
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
    isnothing(temp) || apply_bc_temp!(x.temp, t, setup)

    create_stepper(method; setup, psolver, x.u, x.temp, t, n = n + 1)
end

timestep(method::LMWray3, stepper, Δt; θ = nothing) = error("Not yet implemented")
