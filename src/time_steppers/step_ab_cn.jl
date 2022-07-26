function step(stepper::AdamsBashforthCrankNicolsonStepper, Δt)
    (; method, V, p, t, n, cₙ, tₙ, setup, cache, momentum_cache) = stepper
    (; viscosity_model, force, grid, operators, pressure_solver) = setup
    (; NV, Ω⁻¹) = grid
    (; G, M) = operators
    (; Diff) = operators
    (; p_add_solve, α₁, α₂, θ) = method

    Δt ≈ t - tₙ || error("AB-CN-method requires constant time step")

    # For the first time step, this might be necessary
    cₙ = convection(Vₙ, setup)

    # Advance one step
    Vₙ = V
    pₙ = p
    tₙ = t
    cₙ₋₁ = cₙ

    # Evaluate boundary conditions and force at starting point
    bₙ = bodyforce(force, tₙ, setup)

    # Convection of current solution
    cₙ = convection(Vₙ, setup, momentum_cache)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    bₙ₊₁ = bodyforce(force, tₙ + Δt, setup)

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    b = @. (1 - θ) * bₙ + θ * bₙ₊₁

    Gpₙ = G * pₙ
    d = Diff * V

    # Right hand side of the momentum equation update
    Rr = @. Vₙ + Ω⁻¹ * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + b - Gpₙ)

    # Implicit time-stepping for diffusion
    if viscosity_model isa LaminarModel
        # Use precomputed LU decomposition
        V = Diff_fact \ Rr
    else
        # Get `∇d` since `Diff` is not constant
        ∇d = diffusion_jacobian(V, t, setup)
        V = ∇d \ Rr
    end

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V) / Δt

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(pressure_solver, f)

    # Update velocity field
    V = V .- Δt .* Ω⁻¹ .* (G * Δp)

    # First order pressure:
    p = pₙ + Δp

    if p_add_solve
        # Momentum already contains G*p with the current p, we therefore
        # effectively solve for the pressure difference
        F = momentum(V, p, tₙ + Δt, setup)
        f = M * (Ω⁻¹ .* F)
        Δp = pressure_poisson(pressure_solver, f)
        p = p + Δp
    end

    n = n + 1
    t = tₙ + Δt

    AdamsBashforthCrankNicolsonStepper(; method, V, p, t, n, cₙ, tₙ, setup)
end

"""
    step!(stepper::AdamsBashforthCrankNicolsonStepper, Δt)

Perform one time step with Adams-Bashforth for convection and Crank-Nicolson for diffusion.

`cₙ₋₁` are the convection terms of `tₙ₋₁`. Output includes convection terms at `tₙ`, which
will be used in next time step in the Adams-Bashforth part of the method.

Adams-Bashforth for convection and Crank-Nicolson for diffusion
formulation:

```math
\\begin{align*}
(\\mathbf{u}^{n+1} - \\mathbf{u}^n) / Δt & =
    -(\\alpha_1 \\mathbf{c}^n + \\alpha_2 \\mathbf{c}^{n-1}) \\\\
    & + \\theta \\mathbf{d}^{n+1} + (1-\\theta) \\mathbf{d}^n \\\\
    & + \\theta \\mathbf{F}^{n+1} + (1-\\theta) \\mathbf{F}^n \\\\
    & + \\theta \\mathbf{BC}^{n+1} + (1-\\theta) \\mathbf{BC}^n \\\\
    & - \\mathbf{G} \\mathbf{p} + \\mathbf{y}_p
\\end{align*}
```

where BC are boundary conditions of diffusion. This is rewritten as:

```math
\\begin{align*}
(\\frac{1}{\\Delta t} \\mathbf{I} - \\theta \\mathbf{D}) \\mathbf{u}^{n+1} & =
    (\\frac{1}{\\Delta t} \\mathbf{I} - (1 - \\theta) \\mathbf{D}) \\mathbf{u}^{n} \\\\
    & - (\\alpha_1 \\mathbf{c}^n + \\alpha_2 \\mathbf{c}^{n-1}) \\\\
    & + \\theta \\mathbf{F}^{n+1} + (1-\\theta) \\mathbf{F}^{n} \\\\
    & + \\theta \\mathbf{BC}^{n+1} + (1-\\theta) \\mathbf{BC}^{n} \\\\
    & - \\mathbf{G} \\mathbf{p} + \\mathbf{y}_p
\\end{align*}
```

The LU decomposition of the LHS matrix is precomputed in `operator_convection_diffusion.jl`.

Note that, in constrast to explicit methods, the pressure from previous time steps has an
influence on the accuracy of the velocity.
"""
function step!(stepper::AdamsBashforthCrankNicolsonStepper, Δt)
    (; method, V, p, t, n, cₙ, tₙ, setup, cache, momentum_cache) = stepper
    (; viscosity_model, force, grid, operators, pressure_solver) = setup
    (; NV, Ω⁻¹) = grid
    (; G, M) = operators
    (; Diff) = operators
    (; p_add_solve, α₁, α₂, θ) = method
    (; Vₙ, pₙ, cₙ₋₁, F, f, Δp, Rr, b, bₙ, bₙ₊₁, Gpₙ, Diff_fact) = cache
    (; d, ∇d) = momentum_cache

    Δt ≈ t - tₙ || error("AB-CN-method requires constant time step")

    # For the first time step, this might be necessary
    convection!(cₙ, Vₙ, setup, momentum_cache)

    # Advance one step
    Vₙ .= V
    pₙ .= p
    tₙ = t
    cₙ₋₁ .= cₙ

    # Evaluate boundary conditions and force at starting point
    bodyforce!(force, bₙ, tₙ, setup)

    # Convection of current solution
    convection!(cₙ, Vₙ, setup, momentum_cache)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    bodyforce!(force, bₙ₊₁, tₙ + Δt, setup)

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    @. b = (1 - θ) * bₙ + θ * bₙ₊₁

    mul!(Gpₙ, G, pₙ)
    mul!(d, Diff, V)

    # Right hand side of the momentum equation update
    @. Rr = Vₙ + Ω⁻¹ * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + b - Gpₙ)

    # Implicit time-stepping for diffusion
    if viscosity_model isa LaminarModel
        # Use precomputed LU decomposition
        ldiv!(V, Diff_fact, Rr)
    else
        # Get `∇d` since `Diff` is not constant
        diffusion_jacobian!(∇d, V, t, setup)
        V .= ∇d \ Rr
    end

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V) / Δt

    # Solve the Poisson equation for the pressure
    pressure_poisson!(pressure_solver, Δp, f)

    # Update velocity field
    V .= V .- Δt .* Ω⁻¹ .* (G * Δp)

    # First order pressure:
    p .= pₙ .+ Δp

    if p_add_solve
        # Momentum already contains G*p with the current p, we therefore
        # effectively solve for the pressure difference
        momentum!(F, V, p, tₙ + Δt, setup, momentum_cache)
        @. F = Ω⁻¹ .* F
        mul!(f, M, F)
        pressure_poisson!(pressure_solver, Δp, f)
        p .= p .+ Δp
    end

    n = n + 1
    t = tₙ + Δt

    AdamsBashforthCrankNicolsonStepper(; method, V, p, t, n, cₙ, tₙ, setup)
end
