"""
    momentum(V, ϕ, p, t, setup; getJacobian = false, nopressure = false)

Convenience function for initializing arrays `F` and `∇F` before filling in momentum terms.
"""
function momentum(V, ϕ, p, t, setup; getJacobian = false, nopressure = false)
    @unpack NV = setup.grid

    cache = MomentumCache(setup)
    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    momentum!(F, ∇F, V, ϕ, p, t, setup, cache; getJacobian)
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
    F,
    ∇F,
    V,
    ϕ,
    p,
    t,
    setup,
    cache;
    getJacobian = false,
    nopressure = false,
)
    @unpack Nu, Nv, NV, indu, indv = setup.grid
    @unpack Gx, Gy, y_px, y_py = setup.discretization

    # Store intermediate results in temporary variables
    @unpack c, ∇c, d, ∇d, b, ∇b, Gp = cache

    Gpx = @view Gp[indu]
    Gpy = @view Gp[indv]
    Fx = @view F[indu]
    Fy = @view F[indv]

    # Unsteady BC
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    # Convection
    convection!(c, ∇c, V, ϕ, t, setup, cache, getJacobian)

    # Diffusion
    diffusion!(d, ∇d, V, t, setup, getJacobian)

    # Body force
    bodyforce!(b, ∇b, V, t, setup, getJacobian)

    # Residual in Finite Volume form, including the pressure contribution
    @. F = -c + d + b

    # Nopressure = false is the most common situation, in which we return the entire
    # Right-hand side vector
    if !nopressure
        mul!(Gpx, Gx, p)
        mul!(Gpy, Gy, p)
        @. Fx -= Gpx + y_px
        @. Fy -= Gpy + y_py
    end

    if getJacobian
        # Jacobian requested
        # We return only the Jacobian with respect to V (not p)
        @. ∇F = -∇c + ∇d + ∇b
    end

    F, ∇F
end
