create_stepper(::ExplicitRungeKuttaMethod; setup, psolver, u, temp, t, n = 0) =
    (; setup, psolver, u, temp, t, n)

function timestep!(method::ExplicitRungeKuttaMethod, stepper, Δt; θ = nothing, cache)
    (; setup, psolver, u, temp, t, n) = stepper
    (; closure_model, temperature) = setup
    (; A, b, c) = method
    (; ustart, ku, p, tempstart, ktemp, diff) = cache
    nstage = length(b)
    m = closure_model

    # Update current solution
    tstart = t
    copyto!(ustart, u)
    isnothing(temp) || copyto!(tempstart, temp)

    for i = 1:nstage
        # Compute force at current stage i
        apply_bc_u!(u, t, setup)
        isnothing(temp) || apply_bc_temp!(temp, t, setup)
        momentum!(ku[i], u, temp, t, setup)
        if !isnothing(temp)
            ktemp[i] .= 0
            convection_diffusion_temp!(ktemp[i], u, temp, setup)
            temperature.dodissipation && dissipation!(ktemp[i], diff, u, setup)
        end

        # Add closure term
        isnothing(m) || (ku[i] .+= m(u, θ))

        # Intermediate time step
        t = tstart + c[i] * Δt

        # Apply stage forces
        u .= ustart
        for j = 1:i
            @. u += Δt * A[i, j] * ku[j]
        end
        if !isnothing(temp)
            temp .= tempstart
            for j = 1:i
                @. temp += Δt * A[i, j] * ktemp[j]
            end
        end

        # Project stage u directly
        # Make velocity divergence free at time t
        apply_bc_u!(u, t, setup)
        project!(u, setup; psolver, p)
    end

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    apply_bc_u!(u, t, setup)
    isnothing(temp) || apply_bc_temp!(temp, t, setup)

    create_stepper(method; setup, psolver, u, temp, t, n = n + 1)
end

function timestep(method::ExplicitRungeKuttaMethod, stepper, Δt; θ = nothing)
    (; setup, psolver, u, temp, t, n) = stepper
    (; closure_model, temperature) = setup
    (; A, b, c) = method
    nstage = length(b)
    m = closure_model

    # Update current solution (does not depend on previous step size)
    tstart = t
    ustart = u
    ku = ()
    ktemp = ()

    for i = 1:nstage
        # Compute force at current stage i
        u = apply_bc_u(u, t, setup)
        isnothing(temp) || (temp = apply_bc_temp(temp, t, setup))
        F = momentum(u, temp, t, setup)
        if !isnothing(temp)
            Ftemp = convection_diffusion_temp(u, temp, setup)
            temperature.dodissipation && (Ftemp += dissipation(u, setup))
        end

        # Add closure term
        isnothing(m) || (F = F .+ m(u, θ))

        # Store right-hand side of stage i
        ku = (ku..., F)
        isnothing(temp) || (ktemp = (ktemp..., Ftemp))

        # Intermediate time step
        t = tstart + c[i] * Δt

        # Apply stage forces
        u = ustart
        for j = 1:i
            u = @. u + Δt * A[i, j] * ku[j]
        end
        if !isnothing(temp)
            temp = tempstart
            for j = 1:i
                temp = @. temp + Δt * A[i, j] * ktemp[j]
            end
        end

        # Project stage u directly
        # Make velocity divergence free at time t
        u = apply_bc_u(u, t, setup)
        u = project(u, setup; psolver)
    end

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    u = apply_bc_u(u, t, setup)
    isnothing(temp) || (temp = apply_bc_temp(temp, t, setup))

    create_stepper(method; setup, psolver, u, temp, t, n = n + 1)
end
