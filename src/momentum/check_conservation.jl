"""
    check_conservation(V, t, setup)

Check mass, momentum and energy conservation properties of velocity field.
"""
function check_conservation(V, t, setup)
    @unpack M, yM = setup.discretization
    @unpack indu, indv, Ω, x, y, xp, yp, hx, hy, gx, gy = setup.grid
    @unpack u_bc, v_bc = setup.bc

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    Ωu = @view Ω[indu]
    Ωv = @view Ω[indv]

    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, t)
    end

    uLe_i = u_bc.(x[1], yp, t, [setup])
    uRi_i = u_bc.(x[end], yp, t, [setup])
    vLo_i = v_bc.(xp, y[1], t, [setup])
    vUp_i = v_bc.(xp, y[end], t, [setup])

    # Check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V + yM))

    # Calculate total momentum
    umom = sum(Ωu .* uₕ)
    vmom = sum(Ωv .* vₕ)

    # Add boundary contributions in case of Dirichlet BC
    if setup.bc.u.left == :dirichlet
        umom += sum(uLe_i .* hy) * gx[1]
    end
    if setup.bc.u.right == :dirichlet
        umom += sum(uRi_i .* hy) * gx[end]
    end
    if setup.bc.v.low == :dirichlet
        vmom += sum(vLo_i .* hx) * gy[1]
    end
    if setup.bc.v.up == :dirichlet
        vmom += sum(vUp_i .* hx) * gy[end]
    end

    # Calculate total kinetic energy
    k = 1 / 2 * sum(Ω .* V .^ 2)

    # Add boundary contributions in case of Dirichlet BC
    if setup.bc.u.left == :dirichlet
        k += 1 / 2 * sum(uLe_i .^ 2 .* hy) * gx[1]
    end
    if setup.bc.u.right == :dirichlet
        k += 1 / 2 * sum(uRi_i .^ 2 .* hy) * gx[end]
    end
    if setup.bc.v.low == :dirichlet
        k += 1 / 2 * sum(vLo_i .^ 2 .* hx) * gy[1]
    end
    if setup.bc.v.up == :dirichlet
        k += 1 / 2 * sum(vUp_i .^ 2 .* hx) * gy[end]
    end

    maxdiv, umom, vmom, k
end
