"""
    step!(stepper::ImplicitRungeKuttaStepper, Δt)

Do one time step for implicit Runge-Kutta method.

Unsteady Dirichlet boundary points are not part of solution vector but
are prescribed in a "strong" manner via the `u_bc` and `v_bc` functions.
"""
function step!(stepper::ImplicitRungeKuttaStepper, Δt; cache, momentum_cache)
    (; method, V, p, t, n, setup, pressure_solver) = stepper
    (; grid, operators) = setup
    (; NV, Np, Ω⁻¹) = grid
    (; G, M) = operators
    (; p_add_solve, maxiter, abstol, newton_type) = method
    (; Vtotₙ, ptotₙ, Qⱼ, Fⱼ, ∇Fⱼ, fⱼ, F, ∇F, f, Δp, Gp) = cache
    (; Mtot, Ωtot, dfmom, Z) = cache
    (; A, b, c, Ω_sNV, A_ext, b_ext, Vₙ, pₙ, tₙ) = cache

    is_patterned = stepper.n > 1

    # Update current solution (does not depend on previous step size)
    Vₙ .= V
    pₙ .= p
    tₙ = t

    # Number of stages
    s = length(b)

    # Time instances at all stages, tⱼ = [t₁, t₂, ..., tₛ]
    tⱼ = @. tₙ + c * Δt

    Vtotₙ_mat = reshape(Vtotₙ, :, s)
    ptotₙ_mat = reshape(ptotₙ, :, s)
    for i = 1:s
        # Initialize with the solution at tₙ
        Vtotₙ_mat[:, i] .= Vₙ
        ptotₙ_mat[:, i] .= pₙ
    end

    # Iteration counter
    iter = 0

    # Index in global solution vector
    ind_Vⱼ = 1:(NV*s)
    ind_pⱼ = (NV*s+1):((NV+Np)*s)

    # Vtot contains all stages and is ordered as [u₁; v₁; u₂; v₂; ...; uₛ; vₛ];
    # Starting guess for intermediate stages
    # This can be improved, see e.g. the Radau, Gauss4, or Lobatto scripts
    Vⱼ = @view Qⱼ[ind_Vⱼ]
    pⱼ = @view Qⱼ[ind_pⱼ]
    Vⱼ .= Vtotₙ
    pⱼ .= ptotₙ

    # Initialize right-hand side for all stages
    momentum_allstage!(Fⱼ, ∇Fⱼ, Vⱼ, pⱼ, tⱼ, setup, cache, momentum_cache)

    fmomⱼ = @view fⱼ[ind_Vⱼ]
    fmassⱼ = @view fⱼ[ind_pⱼ]

    # Initialize momentum residual
    mul!(fmomⱼ, A_ext, Fⱼ)
    @. fmomⱼ -= (Ωtot * Vⱼ - Ωtot * Vtotₙ) / Δt
    # fmomⱼ .= .-(Ωtot .* Vⱼ .- Ωtot .* Vtotₙ) ./ Δt .+ A_ext * Fⱼ

    # Initialize mass residual
    mul!(fmassⱼ, Mtot, Vⱼ, -1, -1)
    # fmassⱼ .= .-(Mtot * Vⱼ)

    if newton_type == :approximate
        # Approximate Newton (Jacobian is based on current solution Vₙ)
        momentum!(F, ∇F, Vₙ, pₙ, tₙ, setup, momentum_cache; getJacobian = true)

        # Update iteration matrix, which is now fixed during iterations
        if is_patterned
            kron!(dfmom, A, ∇F)
        else
            dfmom .= kron(A, ∇F)
        end
        @. dfmom = Ω_sNV / Δt - dfmom
        # dfmom .= Ω_sNV ./ Δt .- kron(A, ∇F)
        Z[ind_Vⱼ, ind_Vⱼ] .= dfmom

        # Determine LU decomposition
        # Z_fact = factorize(Z)
        Z_fact = lu(Z)
    end

    while maximum(abs.(f)) > abstol
        if newton_type == :approximate
            # Approximate Newton
            # ΔQⱼ = Z \ fⱼ

            # Re-use the decomposition
            # ΔQⱼ = Z_fact \ fⱼ
            ldiv!(ΔQⱼ, Z_fact, fⱼ)
        elseif newton_type == :full
            # Full Newton
            momentum_allstage!(
                Fⱼ,
                ∇Fⱼ,
                Vⱼ,
                pⱼ,
                tⱼ,
                setup,
                cache,
                momentum_cache;
                getJacobian = true,
            )

            # Update iteration matrix
            mul!(dfmom, A_ext, ∇Fⱼ)
            dfmom = Ω_sNV / Δt - dfmom
            # dfmom .= Ω_sNV / Δt - A_ext * ∇Fⱼ
            Z[ind_Vⱼ, ind_Vⱼ] .= dfmom

            # Get change
            ΔQⱼ = Z \ fⱼ
        end

        # Update solution vector
        Qⱼ .+= ΔQⱼ

        # Update iteration counter
        iter += 1

        # Evaluate RHS for next iteration and check residual based on computed Vⱼ, pⱼ
        momentum_allstage!(Fⱼ, ∇Fⱼ, Vⱼ, pⱼ, tⱼ, setup, cache, momentum_cache)
        mul!(fmomⱼ, A_ext, Fⱼ)
        @. fmomⱼ -= (Ωtot * Vⱼ - Ωtot * Vtotₙ) / Δt
        # fmomⱼ = -(Ωtot .* Vⱼ - Ωtot .* Vtotₙ) / Δt + A_ext * Fⱼ
        mul!(fmassⱼ, Mtot, Vⱼ, -1, -1)
        # fmassⱼ = -(Mtot * Vⱼ)

        iter ≤ maxiter || error("Newton solver not converged in $maxiter iterations")
    end

    # Solution at new time step with b-coefficients of RK method
    mul!(V, b_ext, Fⱼ)
    @. V = Vₙ + Δt * Ω⁻¹ * V
    # V .= Vₙ .+ Δt .* Ω⁻¹ .* (b_ext * Fⱼ)

    # Make V satisfy the incompressibility constraint at n+1; this is only needed when the
    # boundary conditions are time-dependent. For stiffly accurate methods, this can also
    # be skipped (e.g. Radau IIA) - this still needs to be implemented

    # For steady bc we do an additional pressure solve
    # That saves a pressure solve for iter = 1 in the next time step
    if p_add_solve
        # Momentum already contains G*p with the current p, we therefore
        # effectively solve for the pressure difference
        momentum!(F, V, p, tₙ + Δt, setup, momentum_cache)
        @. F = Ω⁻¹ .* F
        mul!(f, M, F)
        pressure_poisson!(pressure_solver, Δp, f)
        p .= p .+ Δp
    end

    # Standard method; take pressure of last stage
    p .= pⱼ[(end-Np+1):end]

    ImplicitRungeKuttaStepper(; method, V, p, t, n, setup)
end

"""
    momentum_allstage!(F, ∇F, V, p, t, setup, cache, momentum_cache; getJacobian = false)

Call momentum for multiple `(V, p)` pairs, as required in implicit RK methods.
"""
function momentum_allstage!(
    Fⱼ,
    ∇Fⱼ,
    Vⱼ,
    pⱼ,
    tⱼ,
    setup,
    cache,
    momentum_cache;
    getJacobian = false,
)
    (; NV, Np) = setup.grid
    (; c, ∇F) = cache

    for i = 1:length(c)
        # Indices for current stage
        ind_Vᵢ = (1:NV) .+ NV * (i - 1)
        ind_pᵢ = (1:Np) .+ Np * (i - 1)

        # Quantities at current stage
        F = @view Fⱼ[ind_Vᵢ]
        V = @view Vⱼ[ind_Vᵢ]
        p = @view pⱼ[ind_pᵢ]
        t = tⱼ[i]

        # Compute residual and Jacobian for this stage
        momentum!(F, ∇F, V, p, t, setup, momentum_cache; getJacobian)

        ∇Fⱼ[ind_Vᵢ, ind_Vᵢ] = ∇F
    end

    Fⱼ, ∇Fⱼ
end
