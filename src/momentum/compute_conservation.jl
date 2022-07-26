"""
    compute_conservation(V, setup)

Compute mass, momentum and energy conservation properties of velocity field.
"""
function compute_conservation end

# 2D version
function compute_conservation(V, setup::Setup{T,2}) where {T}
    (; M) = setup.operators
    (; indu, indv, Ω) = setup.grid

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    Ωu = @view Ω[indu]
    Ωv = @view Ω[indv]

    # Check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V))

    # Calculate total momentum
    umom = sum(Ωu .* uₕ)
    vmom = sum(Ωv .* vₕ)

    # Calculate total kinetic energy
    k = 1 / 2 * sum(Ω .* V .^ 2)

    maxdiv, umom, vmom, k
end

# 3D version
function compute_conservation(V, setup::Setup{T,3}) where {T}
    (; M) = setup.operators
    (; indu, indv, indw, Ω) = setup.grid

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    Ωu = @view Ω[indu]
    Ωv = @view Ω[indv]
    Ωw = @view Ω[indw]

    # Check if new velocity field is divergence free (mass conservation)
    maxdiv = maximum(abs.(M * V))

    # Calculate total momentum
    umom = sum(Ωu .* uₕ)
    vmom = sum(Ωv .* vₕ)
    wmom = sum(Ωw .* wₕ)

    # Calculate total kinetic energy
    k = 1 / 2 * sum(Ω .* V .^ 2)

    maxdiv, umom, vmom, wmom, k
end
