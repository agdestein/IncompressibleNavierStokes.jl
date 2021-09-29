"""
    bodyforce!(F, ∇F V, t, setup, getJacobian)

Compute body force `F` in momentum equations at velocity points.
If `getJacobian`, also compute `∇F = dF/dV`.

If
"""
function bodyforce!(F, ∇F, V, t, setup, getJacobian = false)
    @unpack indu, indv = setup.grid

    if setup.force.force_unsteady
        Fx, ∇Fx = setup.force.bodyforce_x.(xu, yu, t, [setup], getJacobian)
        Fy, ∇Fy = setup.force.bodyforce_y.(xv, yv, t, [setup], getJacobian)

        # this works for both 2nd and 4th order method
        Fy = -Omv .* Fy[:]

        F[indu] .= Fx
        F[indv] .= Fy
        if getJacobian
            ∇F[indu, :]  = ∇Fx
            ∇F[indv, :]  = ∇Fy
        end
    else
        Fx = setup.force.Fx;
        Fy = setup.force.Fy;

        getJacobian && (∇F .= 0)
    end

    F, ∇F
end

"""
Fx, Fy, dFx, dFy = force(V, t, setup, getJacobian)
Body force in momentum equations in Finite Volume setting, so integrated dFx, dFy are the Jacobians dFx/dV and dFy/dV
"""
function bodyforce(V, t, setup, getJacobian = false)
    # Fx, dFx = setup.force.bodyforce_x.(xu, yu, t, [setup], getJacobian)
    # Fy, dFy = setup.force.bodyforce_y.(xv, yv, t, [setup], getJacobian)

    # # this works for both 2nd and 4th order method
    # Fy = -Omv .* Fy[:]

    @unpack Nu, Nv = setup.grid

    Fx = zeros(Nu)
    Fy = zeros(Nv)
    dFx = spzeros(Nu, Nu + Nv)
    dFy = spzeros(Nv, Nu + Nv)

    Fx, Fy, dFx, dFy
end
