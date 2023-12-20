create_stepper(::ExplicitRungeKuttaMethod; setup, psolver, u, p, t, n = 0) =
    (; setup, psolver, u, p, t, n)

function timestep!(method::ExplicitRungeKuttaMethod, stepper, Δt; cache, θ = nothing)
    (; setup, psolver, u, p, t, n) = stepper
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    (; A, b, c, p_add_solve) = method
    (; u₀, ku, v, F, M, G) = cache

    D = dimension()

    # Update current solution (does not depend on previous step size)
    t₀ = t
    for α = 1:D
        u₀[α] .= u[α]
    end

    # Number of stages
    nstage = length(b)

    ## Start looping over stages

    # At i = 1 we calculate F₁ = F(u₀), p₁ and u₁
    # ⋮
    # At i = s we calculate Fₛ = F(uₛ₋₁), pₛ, and uₛ
    for i = 1:nstage
        # Right-hand side for tᵢ₋₁ based on current velocity field uᵢ₋₁, vᵢ₋₁ at
        # level i-1. This includes force evaluation at tᵢ₋₁.
        momentum!(F, u, t, setup; θ)

        # Store right-hand side of stage i
        for α = 1:D
            @. ku[i][α] = F[α]
        end

        # Intermediate time step
        t = t₀ + c[i] * Δt

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives vᵢ
        for α = 1:D
            v[α] .= u₀[α]
            for j = 1:i
                @. v[α] += Δt * A[i, j] * ku[j][α]
                # @. v[α][Iu[α]] += Δt * A[i, j] * ku[j][α][Iu[α]]
            end
        end

        # Boundary conditions at tᵢ
        apply_bc_u!(v, t, setup)

        # Divergence of tentative velocity field
        divergence!(M, v, setup)
        @. M *= Ω / (c[i] * Δt)

        # Solve the Poisson equation
        poisson!(psolver, p, M)
        apply_bc_p!(p, t, setup)

        # Compute pressure correction term
        pressuregradient!(G, p, setup)

        # Update velocity current stage, which is now divergence free
        for α = 1:D
            @. u[α] = v[α] - c[i] * Δt * G[α]
            # @. u[α][Iu[α]] = v[α][Iu[α]] - c[i] * Δt * G[α][Iu[α]]
        end
        apply_bc_u!(u, t, setup)
    end

    # Complete time step
    t = t₀ + Δt

    # Do additional pressure solve to avoid first order pressure
    p_add_solve && pressure!(psolver, u, p, t, setup, F, G, M)

    create_stepper(method; setup, psolver, u, p, t, n = n + 1)
end

function timestep(method::ExplicitRungeKuttaMethod, stepper, Δt; θ = nothing)
    (; setup, psolver, u, p, t, n) = stepper
    (; grid) = setup
    (; dimension) = grid
    (; A, b, c) = method

    D = dimension()

    # Update current solution (does not depend on previous step size)
    t₀ = t
    u₀ = u

    # Number of stages
    nstage = length(b)

    ku = ()

    ## Start looping over stages

    # At i = 1 we calculate F₁ = F(u₀), p₁ and u₁
    # ⋮
    # At i = s we calculate Fₛ = F(uₛ₋₁), pₛ, and uₛ
    for i = 1:nstage
        # Right-hand side for tᵢ₋₁ based on current velocity field uᵢ₋₁, vᵢ₋₁ at
        # level i-1. This includes force evaluation at tᵢ₋₁.
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

        # Boundary conditions at tᵢ
        u = apply_bc_u(u, t, setup)

        # Make divergence free
        u = project(psolver, u, setup)
    end

    # Complete time step
    t = t₀ + Δt

    create_stepper(method; setup, psolver, u, p, t, n = n + 1)
end
