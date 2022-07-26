"""
    momentum(V, p, t, setup; nopressure = false)

Convenience function for initializing arrays `F` and `∇F` before filling in momentum terms.
"""
function momentum(V, p, t, setup; nopressure = false)
    (; viscosity_model, force) = setup
    (; G) = setup.operators
    c = convection(V, setup)
    d = diffusion(viscosity_model, V, setup)
    b = bodyforce(force, t, setup)
    F = -c + d + b
    nopressure ? F : F - G * p
end

"""
    momentum!(F, V, p, t, setup, cache; nopressure = false)

Calculate rhs of momentum equations and, optionally, Jacobian with respect to velocity field.

  - `V`: velocity field
  - `p`: pressure
  - `nopressure`: exclude pressure gradient; in this case input argument `p` is not used
"""
function momentum!(F, V, p, t, setup, cache; nopressure = false)
    (; viscosity_model, force) = setup
    (; G) = setup.operators

    # Temporary arrays for storing intermediate results
    (; c, d, b, Gp) = cache

    # Forces
    convection!(c, V, setup, cache)
    diffusion!(viscosity_model, d, V, setup)
    bodyforce!(force, b, t, setup)

    # Residual in Finite Volume form, including the pressure contribution
    @. F = -c + d + b

    # nopressure = false is the most common situation, in which we return the entire
    # right-hand side vector
    if !nopressure
        mul!(Gp, G, p)
        @. F -= Gp
    end

    F
end

function momentum_jacobian!(∇F, V, p, t, setup, cache; nopressure = false)
    (; viscosity_model) = setup

    # Temporary arrays for storing intermediate results
    (; ∇c, ∇d) = cache

    # Forces
    convection_jacobian!(∇c, V, setup, cache)
    diffusion_jacobian!(viscosity_model, ∇d, V, setup)

    # Residual in Finite Volume form, including the pressure contribution
    @. ∇F = -∇c + ∇d

    ∇F
end
