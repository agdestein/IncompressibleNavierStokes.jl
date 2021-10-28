"""
    bodyforce!(F, ∇F V, t, setup; getJacobian = false)

Compute body force `F` in momentum equations at velocity points.
If `getJacobian`, also compute `∇F = ∂F/∂V`.
"""
function bodyforce!(F, ∇F, V, t, setup; getJacobian = false)
    @unpack indu, indv, xu, xv, yu, yv = setup.grid

    if setup.force.force_unsteady
        Fx, ∇Fx = setup.force.bodyforce_x.(xu, yu, t, [setup]; getJacobian)
        Fy, ∇Fy = setup.force.bodyforce_y.(xv, yv, t, [setup]; getJacobian)

        # This works for both 2nd and 4th order method
        Fy = -Ωv .* Fy[:]

        F[indu] .= Fx
        F[indv] .= Fy
        if getJacobian
            ∇F[indu, :] = ∇Fx
            ∇F[indv, :] = ∇Fy
        end
    else
        F .= setup.force.F
        getJacobian && (∇F .= 0)
    end

    F, ∇F
end

"""
    bodyforce(V, t, setup; getJacobian = false)

Body force in momentum equations in Finite Volume setting, so integrated `dFx`, `dFy` are the Jacobians `∂Fx/∂V` and `∂Fy/∂V`.
"""
function bodyforce(V, t, setup; getJacobian = false)
    # Fx, dFx = setup.force.bodyforce_x.(xu, yu, t, [setup], getJacobian)
    # Fy, dFy = setup.force.bodyforce_y.(xv, yv, t, [setup], getJacobian)

    # # this works for both 2nd and 4th order method
    # Fy = -Ωv .* Fy[:]

    @unpack NV = setup.grid

    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    bodyforce!(F, ∇F, V, t, setup; getJacobian)
end
