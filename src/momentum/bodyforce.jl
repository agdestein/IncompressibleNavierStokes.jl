"""
    bodyforce!(F, ∇F V, t, setup; getJacobian = false)

Compute body force `F` in momentum equations at velocity points.
If `getJacobian`, also compute `∇F = ∂F/∂V`.
"""
function bodyforce!(F, ∇F, V, t, setup; getJacobian = false)
    (; indu, indv, indw, xu, xv, xw, yu, yv, yw, zu, zv, zw) = setup.grid

    if setup.force isa UnsteadyBodyForce
        Fx, ∇Fx = setup.force.bodyforce_x.(xu, yu, zu, t)
        Fy, ∇Fy = setup.force.bodyforce_y.(xv, yv, zv, t)
        Fz, ∇Fz = setup.force.bodyforce_z.(xw, yw, zw, t)

        F[indu] .= Fx
        F[indv] .= Fy
        F[indw] .= Fz
        if getJacobian
            ∇F[indu, :] = ∇Fx
            ∇F[indv, :] = ∇Fy
            ∇F[indw, :] = ∇Fz
        end
    else
        F .= setup.force.F
        getJacobian && (∇F .= 0)
    end

    F, ∇F
end

"""
    bodyforce(V, t, setup; getJacobian = false)

Body force in momentum equations in Finite Volume setting, so integrated `dFx`, `dFy` are
the Jacobians `∂Fx/∂V` and `∂Fy/∂V`.
"""
function bodyforce(V, t, setup; getJacobian = false)
    (; NV) = setup.grid

    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    bodyforce!(F, ∇F, V, t, setup; getJacobian)
end
