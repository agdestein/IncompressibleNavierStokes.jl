create_stepper(::ExplicitRungeKuttaMethod; setup, psolver, u, t, n = 0) =
    (; setup, psolver, u, t, n)

function timestep!(method::ExplicitRungeKuttaMethod, stepper, Δt; θ = nothing, cache)
    (; setup, psolver, u, p, t, n) = stepper
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    (; A, b, c, p_add_solve) = method
    (; u₀, ku, div, p) = cache
    D = dimension()
    nstage = length(b)

    # Update current solution (does not depend on previous step size)
    t₀ = t
    copyto!.(u₀, u)

    for i = 1:nstage
        # Right-hand side for tᵢ₋₁ based on current velocity field uᵢ₋₁, vᵢ₋₁ at
        # level i-1. This includes force evaluation at tᵢ₋₁.
        momentum!(ku[i], u, t, setup; θ)

        # Intermediate time step
        t = t₀ + c[i] * Δt

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives vᵢ
        for α = 1:D
            u[α] .= u₀[α]
            for j = 1:i
                @. u[α] += Δt * A[i, j] * ku[j][α]
                # @. u[α][Iu[α]] += Δt * A[i, j] * ku[j][α][Iu[α]]
            end
        end

        # Make velocity divergence free at time tᵢ
        apply_bc_u!(u, t, setup)
        project!(u, p, setup; psolver, div, p)
    end

    create_stepper(method; setup, psolver, u, t, n = n + 1)
end

function timestep(method::ExplicitRungeKuttaMethod, stepper, Δt; θ = nothing)
    (; setup, psolver, u, p, t, n) = stepper
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
        u = apply_bc_u(u, t, setup)
        F = momentum(u, t, setup; θ)

        # Store right-hand side of stage i
        ku = (ku..., F)

        # Intermediate time step
        t = t₀ + c[i] * Δt

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives vᵢ
        u = u₀
        for j = 1:i
            u = @. u + Δt * A[i, j] * ku[j]
            # u = tupleadd(u, @.(Δt * A[i, j] * ku[j]))
        end

        # Make velocity divergence free at time t
        u = apply_bc_u(u, t, setup)
        u = project(psolver, u, setup)
    end

    create_stepper(method; setup, psolver, u, p, t, n = n + 1)
end
