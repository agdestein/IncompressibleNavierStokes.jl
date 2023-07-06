"""
    compute_conservation(V, t, setup; bc_vectors = nothing)

Compute mass, momentum and energy conservation properties of velocity field.
"""
function compute_conservation end

compute_conservation(V, t, setup; bc_vectors = nothing) =
    compute_conservation(setup.grid.dimension, V, t, setup; bc_vectors = nothing)

# 2D version
function compute_conservation(::Dimension{2}, V, t, setup; bc_vectors = nothing)
    (; grid, operators, boundary_conditions) = setup
    (; indu, indv, Ω, x, y, xp, yp, hx, hy, gx, gy) = grid
    (; M) = operators
    (; bc_unsteady, u_bc, v_bc) = boundary_conditions

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    Ωu = @view Ω[indu]
    Ωv = @view Ω[indv]

    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, t)
    end
    (; yM) = bc_vectors

    uLe_i = reshape(u_bc.(x[1], yp, t), :)
    uRi_i = reshape(u_bc.(x[end], yp, t), :)
    vLo_i = reshape(v_bc.(xp, y[1], t), :)
    vUp_i = reshape(v_bc.(xp, y[end], t), :)

    # Check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V + yM))

    # Calculate total momentum
    umom = sum(Ωu .* uₕ)
    vmom = sum(Ωv .* vₕ)

    # Add boundary contributions in case of Dirichlet BC
    boundary_conditions.u.x[1] == :dirichlet && (umom += sum(uLe_i .* hy) * gx[1])
    boundary_conditions.u.x[2] == :dirichlet && (umom += sum(uRi_i .* hy) * gx[end])
    boundary_conditions.v.y[1] == :dirichlet && (vmom += sum(vLo_i .* hx) * gy[1])
    boundary_conditions.v.y[2] == :dirichlet && (vmom += sum(vUp_i .* hx) * gy[end])

    # Calculate total kinetic energy
    k = 1 // 2 * sum(Ω .* V .^ 2)

    # Add boundary contributions in case of Dirichlet BC
    boundary_conditions.u.x[1] == :dirichlet &&
        (k += 1 // 2 * sum(uLe_i .^ 2 .* hy) * gx[1])
    boundary_conditions.u.x[2] == :dirichlet &&
        (k += 1 // 2 * sum(uRi_i .^ 2 .* hy) * gx[end])
    boundary_conditions.v.y[1] == :dirichlet &&
        (k += 1 // 2 * sum(vLo_i .^ 2 .* hx) * gy[1])
    boundary_conditions.v.y[2] == :dirichlet &&
        (k += 1 // 2 * sum(vUp_i .^ 2 .* hx) * gy[end])

    maxdiv, umom, vmom, k
end

# 3D version
function compute_conservation(::Dimension{3}, V, t, setup; bc_vectors = nothing)
    (; grid, operators, boundary_conditions) = setup
    (; indu, indv, indw, Ω, x, y, z, xp, yp, zp, hx, hy, hz, gx, gy, gz) = grid
    (; M) = operators
    (; bc_unsteady, u_bc, v_bc, w_bc) = boundary_conditions

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    Ωu = @view Ω[indu]
    Ωv = @view Ω[indv]
    Ωw = @view Ω[indw]

    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, t)
    end
    (; yM) = bc_vectors

    uLe_i = reshape(u_bc.(x[1], yp, zp', t), :)
    uRi_i = reshape(u_bc.(x[end], yp, zp', t), :)
    vLo_i = reshape(v_bc.(xp, y[1], zp', t), :)
    vUp_i = reshape(v_bc.(xp, y[end], zp', t), :)
    wBa_i = reshape(w_bc.(xp, yp', z[1], t), :)
    wFr_i = reshape(w_bc.(xp, yp', z[end], t), :)

    # Check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V + yM))

    # Calculate total momentum
    umom = sum(Ωu .* uₕ)
    vmom = sum(Ωv .* vₕ)
    wmom = sum(Ωw .* wₕ)

    # Add boundary contributions in case of Dirichlet BC
    boundary_conditions.u.x[1] == :dirichlet && (umom += sum(uLe_i .* (hz ⊗ hy)) * gx[1])
    boundary_conditions.u.x[2] == :dirichlet && (umom += sum(uRi_i .* (hz ⊗ hy)) * gx[end])
    boundary_conditions.v.y[1] == :dirichlet && (vmom += sum(vLo_i .* (hz ⊗ hx)) * gy[1])
    boundary_conditions.v.y[2] == :dirichlet && (vmom += sum(vUp_i .* (hz ⊗ hx)) * gy[end])
    boundary_conditions.w.z[1] == :dirichlet && (wmom += sum(wBa_i .* (hy ⊗ hx)) * gz[1])
    boundary_conditions.w.z[2] == :dirichlet && (wmom += sum(wFr_i .* (hy ⊗ hx)) * gz[end])

    # Calculate total kinetic energy
    k = 1 // 2 * sum(Ω .* V .^ 2)

    # Add boundary contributions in case of Dirichlet BC
    boundary_conditions.u.x[1] == :dirichlet &&
        (k += 1 // 2 * sum(uLe_i .^ 2 .* (hz ⊗ hy)) * gx[1])
    boundary_conditions.u.x[2] == :dirichlet &&
        (k += 1 // 2 * sum(uRi_i .^ 2 .* (hz ⊗ hy)) * gx[end])
    boundary_conditions.v.y[1] == :dirichlet &&
        (k += 1 // 2 * sum(vLo_i .^ 2 .* (hz ⊗ hx)) * gy[1])
    boundary_conditions.v.y[2] == :dirichlet &&
        (k += 1 // 2 * sum(vUp_i .^ 2 .* (hz ⊗ hx)) * gy[end])
    boundary_conditions.w.z[1] == :dirichlet &&
        (k += 1 // 2 * sum(wBa_i .^ 2 .* (hy ⊗ hx)) * gz[1])
    boundary_conditions.w.z[2] == :dirichlet &&
        (k += 1 // 2 * sum(wFr_i .^ 2 .* (hy ⊗ hx)) * gz[end])

    maxdiv, umom, vmom, wmom, k
end
