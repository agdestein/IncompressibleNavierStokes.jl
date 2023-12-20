create_stepper(::ExplicitRungeKuttaMethod; setup, psolver, u, t, n = 0) =
    (; setup, psolver, u, t, n)

function timestep!(method::ExplicitRungeKuttaMethod, stepper, Δt; θ = nothing, cache)
    (; setup, psolver, u, t, n) = stepper
    (; grid) = setup
    (; dimension, Iu) = grid
    (; A, b, c) = method
    (; u₀, ku, div, p) = cache
    D = dimension()
    nstage = length(b)

    # Update current solution
    t₀ = t
    copyto!.(u₀, u)

    for i = 1:nstage
        # Compute force at current stage i
        apply_bc_u!(u, t, setup)
        momentum!(ku[i], u, t, setup; θ)

        # Intermediate time step
        t = t₀ + c[i] * Δt

        # Apply stage forces
        for α = 1:D
            u[α] .= u₀[α]
            for j = 1:i
                @. u[α] += Δt * A[i, j] * ku[j][α]
                # @. u[α][Iu[α]] += Δt * A[i, j] * ku[j][α][Iu[α]]
            end
        end

        # Make velocity divergence free at time t
        apply_bc_u!(u, t, setup)
        project!(u, setup; psolver, div, p)
    end

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    apply_bc_u!(u, t, setup)

    create_stepper(method; setup, psolver, u, t, n = n + 1)
end

function timestep(method::ExplicitRungeKuttaMethod, stepper, Δt; θ = nothing)
    (; setup, psolver, u, t, n) = stepper
    (; grid) = setup
    (; dimension) = grid
    (; A, b, c) = method
    D = dimension()
    nstage = length(b)

    # Update current solution (does not depend on previous step size)
    t₀ = t
    u₀ = u
    ku = ()

    for i = 1:nstage
        # Compute force at current stage i
        u = apply_bc_u(u, t, setup)
        F = momentum(u, t, setup; θ)

        # Store right-hand side of stage i
        ku = (ku..., F)

        # Intermediate time step
        t = t₀ + c[i] * Δt

        # Apply stage forces
        u = u₀
        for j = 1:i
            u = @. u + Δt * A[i, j] * ku[j]
            # u = tupleadd(u, @.(Δt * A[i, j] * ku[j]))
        end

        # Make velocity divergence free at time t
        u = apply_bc_u(u, t, setup)
        u = project(u, setup; psolver)
    end

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    u = apply_bc_u(u, t, setup)

    create_stepper(method; setup, psolver, u, t, n = n + 1)
end
