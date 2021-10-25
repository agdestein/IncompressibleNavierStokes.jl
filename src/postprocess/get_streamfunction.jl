function get_streamfunction(V, t, setup)
    # Compute streamfunction from a Poisson equation nabla^2 ψ = -ω

    @unpack u_bc, v_bc = setup.bc
    @unpack indu, indv, Nux_in, Nvx_in, Nx, Ny = setup.grid
    @unpack hx, hy, x, y, xp, yp = setup.grid
    @unpack Wv_vx, Wu_uy = setup.discretization

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    # Boundary values by integrating around domain
    # Start with ψ = 0 at lower left corner

    # U = d ψ / dy; integrate low->up
    if setup.bc.u.x[1] == :dirichlet
        # u1 = interp1(y, uLe, yp);
        u1 = u_bc.(x[1], yp, t, [setup])
    elseif setup.bc.u.x[1] ∈ [:pressure, :periodic]
        u1 = uₕ[1:Nux_in:end]
    end
    ψLe = cumsum(hy .* u1)
    ψUpLe = ψLe[end]
    ψLe = ψLe[1:end-1]

    # V = -d ψ / dx; integrate left->right
    if setup.bc.v.y[2] == :dirichlet
        v1 = v_bc.(xp, y[end], t, [setup])
    elseif setup.bc.v.y[2] == :pressure
        v1 = vₕ[end-Nvx_in+1:end]
    elseif setup.bc.v.y[2] == :periodic
        v1 = vₕ[1:Nvx_in]
    end
    ψUp = ψUpLe .- cumsum(hx .* v1)
    ψUpRi = ψUp[end]
    ψUp = ψUp[1:end-1]

    # U = d ψ / dy; integrate up->lo
    if setup.bc.u.x[2] == :dirichlet
        u2 = u_bc.(x[end], yp, t, [setup])
    elseif setup.bc.u.x[2] == :pressure
        u2 = uₕ[Nux_in:Nux_in:end]
    elseif setup.bc.u.x[2] == :periodic
        u2 = uₕ[1:Nux_in:end]
    end
    ψRi = ψUpRi .- cumsum(hy[end:-1:1] .* u2[end:-1:1])
    ψLoRi = ψRi[end]
    ψRi = ψRi[end-1:-1:1]

    # V = -d ψ / dx; integrate right->left
    if setup.bc.v.y[1] == :dirichlet
        v2 = v_bc.(xp, y[1], t, [setup])
    elseif setup.bc.v.y[1] ∈ [:pressure, :periodic]
        v2 = vₕ[1:Nvx_in]
    end
    ψLo = ψLoRi .+ cumsum(hx[end:-1:1] .* v2[end:-1:1])
    ψLoLe = ψLo[end]
    ψLo = ψLo[end-1:-1:1]

    if abs(ψLoLe) > 1e-12
        @warn "Contour integration of ψ not consistent: $(abs(ψLoLe))"
    end

    # Solve del^2 ψ = -ω
    # Only dirichlet boundary conditions because we calculate streamfunction at
    # Inner points only

    # X-direction
    diag1 = 1 ./ hx .* ones(Nx)
    Q1D = spdiagm(Nx, Nx + 1, 0 => -diag1, 1 => diag1)
    Qx_bc = bc_general(Nx + 1, Nx - 1, 2, :dirichlet, :dirichlet, hx[1], hx[end])
    # Extend to 2D
    Q2Dx = kron(sparse(I, Ny - 1, Ny - 1), Q1D * Qx_bc.B1D)
    yQx =
        kron(sparse(I, Ny - 1, Ny - 1), Q1D * Qx_bc.Btemp) *
        (kron(ψLe .* ones(Ny - 1), Qx_bc.ybc1) + kron(ψRi .* ones(Ny - 1), Qx_bc.ybc2))

    # Y-direction
    diag1 = 1 ./ hy .* ones(Ny)
    Q1D = spdiagm(Ny, Ny + 1, 0 => -diag1, 1 => diag1)
    Qy_bc = bc_general(Ny + 1, Ny - 1, 2, :dirichlet, :dirichlet, hy[1], hy[end])
    # Extend to 2D
    Q2Dy = kron(Q1D * Qy_bc.B1D, sparse(I, Nx - 1, Nx - 1))
    yQy =
        kron(Q1D * Qy_bc.Btemp, sparse(I, Nx - 1, Nx - 1)) *
        (kron(Qy_bc.ybc1, ψLo .* ones(Nx - 1)) + kron(Qy_bc.ybc2, ψUp .* ones(Nx - 1)))

    Aψ = Wv_vx * Q2Dx + Wu_uy * Q2Dy
    yAψ = Wv_vx * yQx + Wu_uy * yQy

    ω = get_vorticity(V, t, setup)
    ω_flat = reshape(ω, length(ω))

    # Solve streamfunction from Poisson equaton
    ψ = -Aψ \ (ω_flat + yAψ)

    reshape(ψ, size(ω)...)
end
