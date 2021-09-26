"""
    Fx, Fy, dFx, dFy = force(V, t, setup, getJacobian)
Body force in momentum equations in Finite Volume setting, so integrated dFx, dFy are the Jacobians dFx/dV and dFy/dV
"""
function force(V, t, setup, getJacobian)
    force_unsteady = setup.case.force_unsteady

    if setup.force.isforce
        # create function handle with name bodyforce
        Fx, Fy, dFx, dFy = setup.force.bodyforce(V, t, setup, getJacobian)
    else
        Nu = setup.grid.Nu
        Nv = setup.grid.Nv
        Fx = zeros(Nu)
        Fy = zeros(Nv)
        dFx = sparse(Nu, Nu + Nv)
        dFy = sparse(Nv, Nu + Nv)

        if force_unsteady
            error("Body force file $file_name not available")
        end
    end

    Fx, Fy, dFx, dFy
end
