function momentum(V, ϕ, p, t, setup, getJacobian = false, nopressure = false)
    @unpack NV = setup.grid

    cache = MomentumCache(setup)
    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    convection!(F, ∇F, V, ϕ, t, setup, cache, getJacobian)
end

"""
    momentum!(F, ∇F, V, ϕ, p, t, setup, cache = MomentumCache(setup), getJacobian = false, nopressure = false)

Calculate rhs of momentum equations and, optionally, Jacobian with respect to velocity field
V: velocity field
ϕ: "convection" field: e.g. d(c_x u)/dx + d(c_y u)/dy; usually c_x = u,
c_y = v, so ϕ = V
p: pressure
getJacobian = true: return ∇FdV
nopressure = true: exclude pressure gradient; in this case input argument p is not used
"""
function momentum!(F, ∇F, V, ϕ, p, t, setup, cache = MomentumCache(setup), getJacobian = false, nopressure = false)
    @unpack Nu, Nv, NV, indu, indv = setup.grid
    @unpack Gx, Gy, y_px, y_py = setup.discretization

    # Store intermediate results in temporary variables
    @unpack c, ∇c, d, ∇d, b, ∇b, Gp = cache

    Gpx = @view Gp[indu]
    Gpy = @view Gp[indv]
    Fx = @view Gp[indu]
    Fy = @view Gp[indv]

    # Unsteady BC
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    # Convection
    convection!(c, ∇c, V, ϕ, t, setup, cache, getJacobian)

    # Diffusion
    diffusion!(d, ∇d, V, t, setup, getJacobian)

    # Body force
    bodyforce!(b, ∇b, V, t, setup, getJacobian);

    # residual in Finite Volume form, including the pressure contribution
    @. F = - c + d + b

    # nopressure = false is the most common situation, in which we return the entire
    # right-hand side vector
    if !nopressure
        mul!(Gpx, Gx, p)
        mul!(Gpy, Gy, p)
        @. Fx -= Gpx + y_px
        @. Fy -= Gpy + y_py
    end

    if getJacobian
        # Jacobian requested
        # we return only the Jacobian with respect to V (not p)
        @. ∇F = - ∇c + ∇d + ∇b
    end

    F, ∇F
end
