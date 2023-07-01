function step(method::AdamsBashforthCrankNicolsonMethod, stepper, Δt; bc_vectors = nothing)
    (; setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ) = stepper
    (; convection_model, viscosity_model, force, grid, operators, boundary_conditions) =
        setup
    (; bc_unsteady) = boundary_conditions
    (; NV, Ω) = grid
    (; G, M) = operators
    (; Diff) = operators
    (; p_add_solve, α₁, α₂, θ) = method

    T = typeof(Δt)

    # For the first time step, this might be necessary
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ)
    end
    cₙ, = convection(convection_model, Vₙ, Vₙ, setup; bc_vectors)

    # Advance one step
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ = V
    pₙ = p
    tₙ = t
    Δtₙ = Δt
    cₙ₋₁ = cₙ
    @assert Δtₙ ≈ Δtₙ₋₁

    # Unsteady BC at current time
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ)
    end
    (; yDiff) = bc_vectors

    yDiffₙ = yDiff

    # Evaluate boundary conditions and force at starting point
    bₙ = force

    # Convection of current solution
    cₙ, = convection(convection_model, Vₙ, Vₙ, setup; bc_vectors)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δt)
    end
    (; yDiff, y_p) = bc_vectors

    bₙ₊₁ = force

    yDiffₙ₊₁ = yDiff

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    b = @. (1 - θ) * bₙ + θ * bₙ₊₁
    yDiff = @. (1 - θ) * yDiffₙ + θ * yDiffₙ₊₁

    Gpₙ = G * pₙ + y_p

    d = Diff * V

    # Right hand side of the momentum equation update
    Rr = @. Vₙ + 1 / Ω * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + yDiff + b - Gpₙ)

    # Implicit time-stepping for diffusion
    if viscosity_model isa LaminarModel
        # Use precomputed LU decomposition
        # FIXME: Put Diff_fact in stepper
        V = Diff_fact \ Rr
    else
        # Get `∇d` since `Diff` is not constant
        d, ∇d = diffusion(V, t, setup; get_jacobian = true)
        V = ∇d \ Rr
    end

    # Make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free (need BC at `tₙ₊₁`)
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δt)
    end
    (; yM) = bc_vectors

    # Boundary condition for Δp between time steps (!= 0 if fluctuating outlet pressure)
    y_Δp = zeros(T, NV)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(pressure_solver, f)

    # Update velocity field
    V -= Δt ./ Ω .* (G * Δp .+ y_Δp)

    # First order pressure:
    p = pₙ .+ Δp

    if p_add_solve
        p = pressure_additional_solve(pressure_solver, V, p, tₙ + Δt, setup; bc_vectors)
    end

    t = tₙ + Δtₙ

    (; method, setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ)
end

function step!(
    method::AdamsBashforthCrankNicolsonMethod,
    stepper,
    Δt;
    cache,
    momentum_cache,
    bc_vectors = nothing,
)
    (; setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ) = stepper
    (; convection_model, viscosity_model, force, grid, operators, boundary_conditions) =
        setup
    (; bc_unsteady) = boundary_conditions
    (; NV, Ω) = grid
    (; G, M) = operators
    (; Diff) = operators
    (; p_add_solve, α₁, α₂, θ) = method
    (; cₙ, cₙ₋₁, F, f, Δp, Rr, b, bₙ, bₙ₊₁, yDiffₙ, yDiffₙ₊₁, Gpₙ, Diff_fact) = cache
    (; d, ∇d) = momentum_cache

    T = typeof(Δt)

    # For the first time step, this might be necessary
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ)
    end
    convection!(convection_model, cₙ, nothing, Vₙ, Vₙ, setup, momentum_cache; bc_vectors)

    # Advance one step
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt
    cₙ₋₁ .= cₙ
    @assert Δtₙ ≈ Δtₙ₋₁

    # Unsteady BC at current time
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ)
    end
    (; yDiff) = bc_vectors

    yDiffₙ .= yDiff

    # Evaluate boundary conditions and force at starting point
    bₙ = force

    # Convection of current solution
    convection!(convection_model, cₙ, nothing, Vₙ, Vₙ, setup, momentum_cache; bc_vectors)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δt)
    end
    (; yDiff, y_p) = bc_vectors
    bₙ₊₁ = force

    yDiffₙ₊₁ .= yDiff

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    @. b = (1 - θ) * bₙ + θ * bₙ₊₁
    yDiff = @. (1 - θ) * yDiffₙ + θ * yDiffₙ₊₁

    mul!(Gpₙ, G, pₙ)
    Gpₙ .+= y_p

    mul!(d, Diff, V)

    # Right hand side of the momentum equation update
    @. Rr = Vₙ + 1 ./ Ω * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + yDiff + b - Gpₙ)

    # Implicit time-stepping for diffusion
    if viscosity_model isa LaminarModel
        # Use precomputed LU decomposition
        if Δt ≉  cache.Δt
            # Time step has changed, recompute LU decomposition
            Diff_fact[] = lu(I(NV) - θ * Δt * Diagonal(1 ./ Ω) * Diff)
            cache.Δt[] = Δt
        end
        ldiv!(V, Diff_fact[], Rr)
    else
        # Get `∇d` since `Diff` is not constant
        diffusion!(d, ∇d, V, t, setup; get_jacobian = true)
        V .= ∇d \ Rr
    end

    # Make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free (need BC at `tₙ₊₁`)
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δt)
    end
    (; yM) = bc_vectors

    # Boundary condition for Δp between time steps (!= 0 if fluctuating outlet pressure)
    y_Δp = zeros(T, NV)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    pressure_poisson!(pressure_solver, Δp, f)

    # Update velocity field
    V .-= Δt ./ Ω .* (G * Δp .+ y_Δp)

    # First order pressure:
    p .= pₙ .+ Δp

    if p_add_solve
        pressure_additional_solve!(
            pressure_solver,
            V,
            p,
            tₙ + Δt,
            setup,
            momentum_cache,
            F,
            f,
            Δp;
            bc_vectors,
        )
    end

    t = tₙ + Δtₙ

    (; method, setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ)
end
