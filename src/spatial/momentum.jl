"""
    momentum(V, C, p, t, setup, getJacobian = false, nopressure = false)

Calculate rhs of momentum equations and, optionally, Jacobian with respect to velocity field
V: velocity field
C: 'convection' field: e.g. d(c_x u)/dx + d(c_y u)/dy; usually c_x = u,
c_y = v, so C = V
p: pressure
getJacobian = true: return dFdV
nopressure = true: exclude pressure gradient; in this case input argument p is not used
"""
function momentum(V, C, p, t, setup, getJacobian = false, nopressure = false)
    @unpack Nu, Nv, nV = setup.grid

    # Unsteady BC
    if setup.bcbc_unsteady
        setup = set_bc_vectors(t, setup)
    end

    if !nopressure
        @unpack Gx, Gy, y_px, y_py = setup.discretization
        Gpx = Gx * p + y_px
        Gpy = Gy * p + y_py
    end

    # Convection
    convu, convv, dconvu, dconvv = convection(V, C, t, setup, getJacobian)

    # Diffusion
    d2u, d2v, dDiffu, dDiffv = diffusion(V, t, setup, getJacobian)

    # Body force
    if setup.force.isforce
        if setup.force.force_unsteady
            Fx, Fy, dFx, dFy = force(V, t, setup, getJacobian);
        else
            Fx = setup.force.Fx;
            Fy = setup.force.Fy;
            dFx = sparse(Nu, NV)
            dFy = sparse(Nv, NV)
        end
    else
        Fx = zeros(Nu);
        Fy = zeros(Nv);
        dFx = sparse(Nu, NV)
        dFy = sparse(Nv, NV)
    end

    # residual in Finite Volume form, including the pressure contribution
    Fu = - convu + d2u + Fx
    Fv = - convv + d2v + Fy

    # nopressure = 0 is the most common situation, in which we return the entire
    # right-hand side vector
    if !nopressure
        Fu -= Gpx
        Fv -= Gpy
    end

    Fres = [Fu; Fv]

    # norm of residual
    maxres = max(abs(Fres))

    if getJacobian
        # Jacobian requested
        # we return only the Jacobian with respect to V (not p)
        dFu = - dconvu + dDiffu + dFx
        dFv = - dconvv + dDiffv + dFy
        dF = [dFu; dFv]
    else
        dF = sparse(Nu + Nv, Nu + Nv)
    end

    maxres, Fres, dF
end
