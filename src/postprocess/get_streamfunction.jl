"""
    get_streamfunction(V, t, setup)

Compute streamfunction ``\\psi`` from a Poisson equation ``\\nabla^2 \\psi = -\\omega``.
"""
function get_streamfunction end

# 2D version
function get_streamfunction(V, t, setup::Setup{T,2}) where {T}
    (; indu, indv, Nux_in, Nvx_in, Nx, Ny) = setup.grid
    (; hx, hy, x, y, xp, yp) = setup.grid
    (; Wv_vx, Wu_uy) = setup.operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    # Boundary values by integrating around domain
    # Start with ψ = 0 at lower left corner

    # u = d ψ / dy; integrate low->up
    u1 = uₕ[1:Nux_in:end]
    ψLe = cumsum(hy .* u1)
    ψUpLe = ψLe[end]
    ψLe = ψLe[1:(end - 1)]

    # v = -d ψ / dx; integrate left->right
    v1 = vₕ[1:Nvx_in]
    ψUp = ψUpLe .- cumsum(hx .* v1)
    ψUpRi = ψUp[end]
    ψUp = ψUp[1:(end - 1)]

    # u = d ψ / dy; integrate up->lo
    u2 = uₕ[1:Nux_in:end]
    ψRi = ψUpRi .- cumsum(hy[end:-1:1] .* u2[end:-1:1])
    ψLoRi = ψRi[end]
    ψRi = ψRi[(end - 1):-1:1]

    # v = -d ψ / dx; integrate right->left
    v2 = vₕ[1:Nvx_in]
    ψLo = ψLoRi .+ cumsum(hx[end:-1:1] .* v2[end:-1:1])
    ψLoLe = ψLo[end]
    ψLo = ψLo[(end - 1):-1:1]

    abs(ψLoLe) > 1e-12 && @warn "Contour integration of ψ not consistent" abs(ψLoLe)


    # Solve del^2 ψ = -ω
    # Only dirichlet boundary conditions because we calculate streamfunction at
    # Inner points only


    # X-direction
    diag = 1 ./ hx
    Q1D = spdiagm(Nx, Nx + 1, 0 => -diag, 1 => diag)
    Qx_bc = bc_general(Nx + 1, Nx - 1, 2, :dirichlet, :dirichlet, hx[1], hx[end])

    # Extend to 2D
    Q2Dx = I(Ny - 1) ⊗ (Q1D * Qx_bc.B1D)
    yQx = (I(Ny - 1) ⊗ (Q1D * Qx_bc.Btemp)) * (ψLe ⊗ Qx_bc.ybc1 + ψRi ⊗ Qx_bc.ybc2)


    # Y-direction
    diag = 1 ./ hy
    Q1D = spdiagm(Ny, Ny + 1, 0 => -diag, 1 => diag)
    Qy_bc = bc_general(Ny + 1, Ny - 1, 2, :dirichlet, :dirichlet, hy[1], hy[end])

    # Extend to 2D
    Q2Dy = (Q1D * Qy_bc.B1D) ⊗ I(Nx - 1)
    yQy = ((Q1D * Qy_bc.Btemp) ⊗ I(Nx - 1)) * (Qy_bc.ybc1 ⊗ ψLo + Qy_bc.ybc2 ⊗ ψUp)


    # FIXME: Dimension error in periodic case
    # @show size(Wv_vx) size(Q2Dx) size(Wu_uy) size(Q2Dy)
    Aψ = Wv_vx * Q2Dx + Wu_uy * Q2Dy
    yAψ = Wv_vx * yQx + Wu_uy * yQy

    ω = get_vorticity(V, t, setup)
    ω_flat = reshape(ω, length(ω))

    # Solve streamfunction from Poisson equation
    ψ = -Aψ \ (ω_flat + yAψ)

    reshape(ψ, size(ω)...)
end

# 3D version
function get_streamfunction(V, t, setup::Setup{T,3}) where {T}
    error("Not implemented")
end
