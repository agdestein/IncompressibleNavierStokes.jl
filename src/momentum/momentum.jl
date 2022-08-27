"""
    momentum(
        V, ϕ, p, t, setup;
        bc_vectors = nothing,
        getJacobian = false,
        nopressure = false,
        newton_factor = false,
    )

Calculate RHS of momentum equations and, optionally, Jacobian with respect to velocity field.

  - `V`: velocity field
  - `ϕ`: convected field: e.g. ``\\frac{\\partial (\\phi_x V)}{\\partial x} + \\frac{\\partial (\\phi_y V)}{\\partial y}``; usually `ϕ = V` (so `ϕx = u`, `ϕy = v`)
  - `p`: pressure
  - `bc_vectors`: boundary condition vectors `y`
  - `getJacobian`: return `∇F = ∂F/∂V`
  - `nopressure`: exclude pressure gradient; in this case input argument `p` is not used
  - `newton_factor`

Non-mutating/allocating/out-of-place version.

See also [`momentum!`](@ref).
"""
function momentum(
    V, ϕ, p, t, setup;
    bc_vectors = nothing,
    getJacobian = false,
    nopressure = false,
    newton_factor = false,
)
    (; viscosity_model, convection_model, force, boundary_conditions, operators) = setup
    (; G) = operators

    # Unsteady BC
    if isnothing(bc_vectors) || boundary_conditions.bc_unsteady
        bc_vectors = get_bc_vectors(setup, t)
    end
    (; y_p) = bc_vectors

    # Convection
    c, ∇c = convection(convection_model, V, ϕ, setup; bc_vectors, getJacobian, newton_factor)

    # Diffusion
    d, ∇d = diffusion(viscosity_model, V, setup; bc_vectors, getJacobian)

    # Body force
    b = bodyforce(force, t, setup)

    # Residual in Finite Volume form, including the pressure contribution
    F = @. -c + d + b

    # Nopressure = false is the most common situation, in which we return the entire
    # right-hand side vector
    if !nopressure
        F = F .- (G * p .+ y_p)
    end

    if getJacobian
        # Jacobian requested
        # We return only the Jacobian with respect to V (not p)
        ∇F = @. -∇c + ∇d
    else
        ∇F = nothing
    end

    F, ∇F
end

"""
    momentum!(F, ∇F, V, ϕ, p, t, setup, cache; getJacobian = false, nopressure = false)

Calculate rhs of momentum equations and, optionally, Jacobian with respect to velocity field.

  - `V`: velocity field
  - `ϕ`: convected field: e.g. ``\\frac{\\partial (\\phi_x V)}{\\partial x} + \\frac{\\partial (\\phi_y V)}{\\partial y}``; usually `ϕ = V` (so `ϕx = u`, `ϕy = v`)
  - `p`: pressure
  - `bc_vectors`: boundary condition vectors `y`
  - `getJacobian`: return `∇F = ∂F/∂V`
  - `nopressure`: exclude pressure gradient; in this case input argument `p` is not used
  - `newton_factor`

Mutating/non-allocating/in-place version.

See also [`momentum`](@ref).
"""
function momentum!(
    F,
    ∇F,
    V,
    ϕ,
    p,
    t,
    setup,
    cache;
    bc_vectors = nothing,
    getJacobian = false,
    nopressure = false,
    newton_factor = false,
)
    (; viscosity_model, convection_model, force, boundary_conditions, operators) = setup
    (; G) = setup.operators

    # Unsteady BC
    if isnothing(bc_vectors) || boundary_conditions.bc_unsteady
        bc_vectors = get_bc_vectors(setup, t)
    end
    (; y_p) = bc_vectors

    # Store intermediate results in temporary variables
    (; c, ∇c, d, ∇d, b, Gp) = cache

    # Convection
    convection!(convection_model, c, ∇c, V, ϕ, setup, cache; bc_vectors, getJacobian, newton_factor)

    # Diffusion
    diffusion!(viscosity_model, d, ∇d, V, setup; bc_vectors, getJacobian)

    # Body force
    bodyforce!(force, b, t, setup)

    # Residual in Finite Volume form, including the pressure contribution
    @. F = -c + d + b

    # Nopressure = false is the most common situation, in which we return the entire
    # right-hand side vector
    if !nopressure
        mul!(Gp, G, p)
        Gp .+= y_p
        @. F -= Gp
    end

    if getJacobian
        # Jacobian requested
        # We return only the Jacobian with respect to V (not p)
        @. ∇F = -∇c + ∇d
    end

    F, ∇F
end
