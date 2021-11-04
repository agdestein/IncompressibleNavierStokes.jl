"""
    momentum(V, ϕ, p, t, setup; getJacobian = false, nopressure = false)

Convenience function for initializing arrays `F` and `∇F` before filling in momentum terms.
"""
function momentum(V, ϕ, p, t, setup; getJacobian = false, nopressure = false)
    @unpack NV = setup.grid

    cache = MomentumCache(setup)
    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    momentum!(F, ∇F, V, ϕ, p, t, setup, cache; getJacobian, nopressure)
end

"""
    momentum!(F, ∇F, V, ϕ, p, t, setup, cache; getJacobian = false, nopressure = false)

Calculate rhs of momentum equations and, optionally, Jacobian with respect to velocity field.

`V`: velocity field
`ϕ`: "convection" field: e.g. `∂(ϕx V)/∂x + ∂(ϕy V)/∂y`; usually `ϕ = V` (so `ϕx = u`, `ϕy = v`)
`p`: pressure
`getJacobian`: return `∇F = ∂F/∂V`
`nopressure`: exclude pressure gradient; in this case input argument `p` is not used
"""
function momentum!(
    F, ∇F, V, ϕ, p, t, setup, cache;
    getJacobian = false,
    nopressure = false,
)
    @unpack NV = setup.grid
    @unpack G, y_p = setup.discretization

    # Store intermediate results in temporary variables
    @unpack c, ∇c, d, ∇d, b, ∇b, Gp = cache

    # Unsteady BC
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    # Convection
    convection!(c, ∇c, V, ϕ, t, setup, cache; getJacobian)

    # Diffusion
    diffusion!(d, ∇d, V, t, setup; getJacobian)

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
