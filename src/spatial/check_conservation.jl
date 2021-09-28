"""
    check_conservation(V, t, setup)

Check mass, momentum and energy conservation properties of velocity field
"""
function check_conservation(V, t, setup)
    @unpack M, yM = setup.discretization
    @unpack Nu, Nv, Omu, Omv, x, y, xp, yp, hx, hy, gx, gy = setup.grid
    @unpack u_bc, v_bc = setup.bc

    uh = V[1:Nu]
    vh = V[Nu+1:Nu+Nv]

    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    uLe_i = u_bc.(x[1], yp, t, [setup])
    uRi_i = u_bc.(x[end], yp, t, [setup])
    vLo_i = v_bc.(xp, y[1], t, [setup])
    vUp_i = v_bc.(xp, y[end], t, [setup])

    # check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V + yM))

    # calculate total momentum
    umom = sum(Omu .* uh)
    vmom = sum(Omv .* vh)

    # add boundary contributions in case of Dirichlet BC
    if setup.bc.u.left == "dir"
        umom += sum(uLe_i .* hy) * gx[1]
    end
    if setup.bc.u.right == "dir"
        umom += sum(uRi_i .* hy) * gx[end]
    end
    if setup.bc.v.low == "dir"
        vmom += sum(vLo_i .* hx) * gy[1]
    end
    if setup.bc.v.up == "dir"
        vmom += sum(vUp_i .* hx) * gy[end]
    end


    # Calculate total kinetic energy (this equals 0.5*(V')*(Omega.*V))
    k = 0.5 * sum(Omu .* uh .^ 2) + 0.5 * sum(Omv .* vh .^ 2)

    # Add boundary contributions in case of Dirichlet BC
    if setup.bc.u.left == "dir"
        k += 0.5 * sum(uLe_i .^ 2 .* hy) * gx[1]
    end
    if setup.bc.u.right == "dir"
        k += 0.5 * sum(uRi_i .^ 2 .* hy) * gx[end]
    end
    if setup.bc.v.low == "dir"
        k += 0.5 * sum(vLo_i .^ 2 .* hx) * gy[1]
    end
    if setup.bc.v.up == "dir"
        k += 0.5 * sum(vUp_i .^ 2 .* hx) * gy[end]
    end

    maxdiv, umom, vmom, k
end
