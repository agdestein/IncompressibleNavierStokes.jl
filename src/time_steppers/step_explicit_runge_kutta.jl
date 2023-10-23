create_stepper(::ExplicitRungeKuttaMethod; setup, pressure_solver, u, p, t, n = 0) =
    (; setup, pressure_solver, u, p, t, n)

function timestep!(method::ExplicitRungeKuttaMethod, stepper, Δt; cache)
    (; setup, pressure_solver, u, p, t, n) = stepper
    (; grid, boundary_conditions) = setup
    (; dimension, Iu, Ip) = grid
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
        momentum!(F, u, t, setup)

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
        @. M = M / (c[i] * Δt)

        # Solve the Poisson equation
        pressure_poisson!(pressure_solver, p, M)
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
    p_add_solve && pressure_additional_solve!(pressure_solver, u, p, t, setup, F, G, M)

    create_stepper(method; setup, pressure_solver, u, p, t, n = n + 1)
end
