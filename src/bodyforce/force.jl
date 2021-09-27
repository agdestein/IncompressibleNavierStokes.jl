"""
    Fx, Fy, dFx, dFy = force(V, t, setup, getJacobian)
Body force in momentum equations in Finite Volume setting, so integrated dFx, dFy are the Jacobians dFx/dV and dFy/dV
"""
function force(V, t, setup, getJacobian)
    if setup.force.isforce
        # create function handle with name bodyforce
        Fx, dFx = setup.force.bodyforce_x.(xu, yu, t, setup, getJacobian)
        Fy, dFy = setup.force.bodyforce_y.(xv, yv, t, setup, getJacobian)

        # this works for both 2nd and 4th order method
        Fy = -Omv .* Fy[:]
    else
        Nu = setup.grid.Nu
        Nv = setup.grid.Nv
        Fx = zeros(Nu)
        Fy = zeros(Nv)
        dFx = sparse(Nu, Nu + Nv)
        dFy = sparse(Nv, Nu + Nv)
        if setup.case.force_unsteady
            error("Unsteady body force not provided")
        end
    end

    Fx, Fy, dFx, dFy
end
