create_stepper(::ExplicitRungeKuttaMethod; setup, pressure_solver, u, p, t, n = 0) =
    (; setup, pressure_solver, u, p, t, n)

function timestep(method::ExplicitRungeKuttaMethod, stepper, Δt)
    (; setup, pressure_solver, u, p, t, n) = stepper
    (; grid, boundary_conditions) = setup
    (; Ω) = grid
    (; A, b, c, p_add_solve) = method

    T = typeof(Δt)

    # Update current solution (does not depend on previous step size)
    n += 1
    Vₙ = u
    pₙ = p
    tₙ = t
    Δtₙ = Δt

    # Number of stages
    nV = length(u)
    np = length(p)
    nstage = length(b)

    # Reset RK arrays
    tᵢ = tₙ
    # kV = zeros(T, nV, 0)
    ku = fill(u, 0)

    ## Start looping over stages

    # At i = 1 we calculate F₁, p₂ and u₂
    # ⋮
    # At i = s we calculate Fₛ, pₙ₊₁, and uₙ₊₁
    for i = 1:nstage
        # Right-hand side for tᵢ based on current velocity field uₕ, vₕ at level i. This
        # includes force evaluation at tᵢ and pressure gradient. Boundary conditions will be
        # set through `get_bc_vectors` inside momentum. The pressure p is not important here,
        # it will be removed again in the next step
        F = momentum(u, p, tᵢ, setup)

        # Store right-hand side of stage i
        # Remove the -G*p contribution (but not y_p)
        kVᵢ = 1 ./ Ω .* (F + G * p)
        # kV = [kV kVᵢ]
        kV = [ku; [kVᵢ]]

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        # V = dot(kV * A[i, 1:i]
        k = sum(A[i, j] * kV[j] for j = 1:i)

        # Boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c[i] * Δtₙ
        if bc_unsteady
            bc_vectors = get_bc_vectors(setup, tᵢ)
        end
        (; yM) = bc_vectors

        # Divergence of intermediate velocity field
        f = (M * (Vₙ / Δtₙ + k) + yM / Δtₙ) / c[i]

        # Solve the Poisson equation, but not for the first step if the boundary conditions are steady
        if boundary_conditions.bc_unsteady || i > 1
            p = pressure_poisson(pressure_solver, f)
        else
            # Bc steady AND i = 1
            p = pₙ
        end

        Gp = G * p

        # Update velocity current stage, which is now divergence free
        V = @. Vₙ + Δtₙ * (k - c[i] / Ω * Gp)
    end

    # For steady bc we do an additional pressure solve
    # That saves a pressure solve for i = 1 in the next time step
    if !bc_unsteady || p_add_solve
        p = pressure_additional_solve(pressure_solver, V, p, tₙ + Δtₙ, setup; bc_vectors)
    end

    t = tₙ + Δtₙ

    create_stepper(method; setup, pressure_solver, bc_vectors, u, p, t, n)
end

function timestep!(method::ExplicitRungeKuttaMethod, stepper, Δt; cache)
    (; setup, pressure_solver, u, p, t, n) = stepper
    (; grid, boundary_conditions) = setup
    (; dimension, Ip) = grid
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
                @. v[α] = v[α] + Δt * A[i, j] * ku[j][α]
            end
        end

        # Boundary conditions at tᵢ
        apply_bc_u!(v, t, setup)

        # Divergence of tentative velocity field
        divergence!(M, v, setup)
        @. M = M / (c[i] * Δt)

        # Solve the Poisson equation
        Min = view(M, Ip)
        pin = view(p, Ip)
        pressure_poisson!(pressure_solver, pin, Min)
        apply_bc_p!(p, t, setup)

        # Compute pressure correction term
        pressuregradient!(G, p, setup)

        # Update velocity current stage, which is now divergence free
        for α = 1:D
            @. u[α] = v[α] - c[i] * Δt * G[α]
        end
        apply_bc_u!(u, tᵢ, setup)
    end

    # Complete time step
    t = t₀ + Δt

    # Do additional pressure solve to avoid first order pressure
    p_add_solve && pressure_additional_solve!(pressure_solver, u, p, t, setup, F, M)

    create_stepper(method; setup, pressure_solver, u, p, t, n = n + 1)
end
