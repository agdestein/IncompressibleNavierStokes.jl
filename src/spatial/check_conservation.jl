"""
    check_conservation(V, t, setup)

Check mass, momentum and energy conservation properties of velocity field
"""
function check_conservation(V, t, setup)
    uh = V(1:Nu)
    vh = V(Nu+1:Nu+Nv)

    bc =  setup.bc
    @unpack u_bc, v_bc = bc

    if setup.bcbc_unsteady
        set_bc_vectors!(setup, t)
    end

    @unpack M, yM = setup.discretization
    @unpack Nu, Nv, Omu, Omv, x, y, xp, yp, hx, hy, gx, gy = setup.grid

    uLe_i = u_bc(x[1], yp, t, setup)
    uRi_i = u_bc(x[end], yp, t, setup)
    vLo_i = v_bc(xp, y[1], t, setup)
    vUp_i = v_bc(xp, y[end], t, setup)

    # check if new velocity field is divergence free (mass conservation)
    maxdiv = max(abs(M * V + yM))

    # calculate total momentum
    umom = sum(Omu .* uh)
    vmom = sum(Omv .* vh)

    # add boundary contributions in case of Dirichlet BC
    if bc.u.left == "dir"
        umom = umom + sum(uLe_i .* hy) * gx[1]
        # 4th order
        # umom[n] = umom[n] + sum(uLe_i .* (α * hy * gx[1] - hy3 * (gx[1] + gx[2])))
    end
    if bc.u.right == "dir"
        umom = umom + sum(uRi_i .* hy) * gx[end]
    end
    if bc.v.low == "dir"
        vmom = vmom + sum(vLo_i .* hx) * gy[1]
    end
    if bc.v.up == "dir"
        vmom = vmom + sum(vUp_i .* hx) * gy[end]
    end


    # Calculate total kinetic energy (this equals 0.5*(V')*(Omega.*V))
    k = 0.5 * sum(Omu .* uh .^ 2) + 0.5 * sum(Omv .* vh .^ 2)

    # Add boundary contributions in case of Dirichlet BC
    if bc.u.left == "dir"
        k += 0.5 * sum((uLe_i .^ 2) .* hy) * gx[1]
    end
    if bc.u.right == "dir"
        k += 0.5 * sum((uRi_i .^ 2) .* hy) * gx[end]
    end
    if bc.v.low == "dir"
        k += 0.5 * sum((vLo_i .^ 2) .* hx) * gy[1]
    end
    if bc.v.up == "dir"
        k += 0.5 * sum((vUp_i .^ 2) .* hx) * gy[end]
    end

    maxdiv, umom, vmom, k
end
