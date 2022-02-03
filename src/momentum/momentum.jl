"""
    momentum(V, ϕ, p, t, setup; getJacobian = false, nopressure = false)

Convenience function for initializing arrays `F` and `∇F` before filling in momentum terms.
"""
function momentum(V, ϕ, p, t, setup; getJacobian = false, nopressure = false)
    (; NV) = setup.grid

    cache = MomentumCache(setup)
    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    momentum!(F, ∇F, V, ϕ, p, t, setup, cache; getJacobian, nopressure)
end

"""
    momentum!(F, ∇F, V, ϕ, p, t, setup, cache; getJacobian = false, nopressure = false)

Calculate rhs of momentum equations and, optionally, Jacobian with respect to velocity field.

- `V`: velocity field
- `ϕ`: convected field: e.g. ``\\frac{\\partial (\\phi_x V)}{\\partial x} + \\frac{\\partial
  (\\phi_y V)}{\\partial y}``; usually `ϕ = V` (so `ϕx = u`, `ϕy = v`)
- `p`: pressure
- `getJacobian`: return `∇F = ∂F/∂V`
- `nopressure`: exclude pressure gradient; in this case input argument `p` is not used
"""
function momentum!(
    F, ∇F, V, ϕ, p, t, setup, cache;
    getJacobian = false,
    nopressure = false,
    newton_factor = false,
)
    (; viscosity_model, convection_model) = setup

    # Unsteady BC (y_p must be loaded after set_bc_vectors!)
    # TODO: preallocate y_p, and only update in set_bc
    setup.bc.bc_unsteady && set_bc_vectors!(setup, t)
    (; G, y_p) = setup.operators

    # Store intermediate results in temporary variables
    (; c, ∇c, d, ∇d, b, ∇b, Gp) = cache

    # Convection
    convection!(convection_model, c, ∇c, V, ϕ, setup, cache; getJacobian, newton_factor)

    # Diffusion
    diffusion!(viscosity_model, d, ∇d, V, setup; getJacobian)

    # Body force
    bodyforce!(b, ∇b, V, t, setup; getJacobian)

    # Residual in Finite Volume form, including the pressure contribution
    @. F = -c + d + b

    # Nopressure = false is the most common situation, in which we return the entire
    # right-hand side vector
    if !nopressure
        mul!(Gp, G, p)
        Gp .+= y_p
        @. F -= Gp
        # F .= F .- (G * p .+ y_p)
    end

    if getJacobian
        # Jacobian requested
        # We return only the Jacobian with respect to V (not p)
        @. ∇F = -∇c + ∇d + ∇b
    end

    F, ∇F
end
