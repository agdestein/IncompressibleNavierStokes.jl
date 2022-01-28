"""
    bodyforce!(F, ∇F V, t, setup; getJacobian = false)

Compute body force `F` in momentum equations at velocity points.
If `getJacobian`, also compute `∇F = ∂F/∂V`.
"""
function bodyforce!(F, ∇F, V, t, setup; getJacobian = false)
    (; indu, indv, indw, xu, xv, xw, yu, yv, yw, zu, zv, zw) = setup.grid

    if setup.force isa UnsteadyBodyForce
        Fx, ∇Fx = setup.force.bodyforce_x.(xu, yu, zu, t, [setup]; getJacobian)
        Fy, ∇Fy = setup.force.bodyforce_y.(xv, yv, zv, t, [setup]; getJacobian)
        Fz, ∇Fz = setup.force.bodyforce_z.(xw, yw, zw, t, [setup]; getJacobian)

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
    # Fx, dFx = setup.force.bodyforce_x.(xu, yu, t, [setup], getJacobian)
    # Fy, dFy = setup.force.bodyforce_y.(xv, yv, t, [setup], getJacobian)

    # # this works for both 2nd and 4th order method
    # Fy = -Ωv .* Fy[:]

    (; NV) = setup.grid

    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    bodyforce!(F, ∇F, V, t, setup; getJacobian)
end
