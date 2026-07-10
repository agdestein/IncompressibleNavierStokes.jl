@testitem "Poiseuille flow with pressure outlet" begin
    # Steady Poiseuille flow: parabolic Dirichlet inflow, no-slip walls,
    # pressure outlet. After integrating to steady state, the volume flux
    # must be conserved along the channel, the interior must be
    # divergence-free, and the profile must match the analytic parabola.
    uin(dim, x, y, t) = dim == 1 ? 6 * y * (1 - y) : zero(x)
    boundary_conditions =
        (; u = ((DirichletBC(uin), PressureBC()), (DirichletBC(), DirichletBC())))
    setup = Setup(; x = (range(0.0, 4.0, 33), range(0.0, 1.0, 17)), boundary_conditions)
    (; Iu, Ip, Δ, xu) = setup

    # Start from the analytic profile and integrate to steady state
    y = reshape(xu[1][2], 1, :)
    u0 = vectorfield(setup)
    u0[:, :, 1] .= 6 .* y .* (1 .- y)
    u0 = apply_bc_u(u0, 0.0, setup)
    state, _ = solve_unsteady(;
        setup,
        tlims = (0.0, 20.0),
        start = (; u = u0),
        Δt = 0.01,
        params = (; viscosity = 0.05),
    )

    # Volume fluxes at the inlet, at interior stations, and at the outlet
    # must all agree
    jrange = Ip.indices[2]
    flux(i) = sum(j -> state.u[i, j, 1] * Δ[2][j], jrange)
    is = Iu[1].indices[1]
    stations = [is[1] - 1; collect(is[1:8:end]); is[end]]
    fluxes = flux.(stations)
    @test maximum(fluxes) - minimum(fluxes) < 1e-8

    # Divergence-free interior
    @test maximum(abs, view(divergence(state.u, setup), Ip)) < 1e-10

    # Parabolic profile at an interior station. The error is dominated by
    # the truncation error of the diffusion stencil in the wall-adjacent
    # half-cells (measured: 2.4e-3 relative at this resolution).
    imid = is[length(is)÷2]
    @test view(state.u, imid, jrange, 1) ≈ uin.(1, 2.0, xu[1][2][jrange], 0.0) rtol = 5e-3
end

@testitem "Time derivative of Dirichlet boundary conditions" begin
    # With `dudt = true`, `apply_bc_u` fills the boundary values with the
    # time derivative of the boundary conditions (approximated internally
    # with a central finite difference in time).
    uin(dim, x, y, t) = dim == 1 ? 1 + 0.5 * sinpi(2t) : zero(x)
    boundary_conditions =
        (; u = ((DirichletBC(uin), PressureBC()), (SymmetricBC(), SymmetricBC())))
    setup = Setup(; x = (range(0.0, 4.0, 17), range(0.0, 1.0, 9)), boundary_conditions)
    (; Iu, Ip) = setup
    u = vectorfield(setup)
    i1 = Iu[1].indices[1][1] - 1 # Inflow boundary layer of normal component
    i2 = Iu[2].indices[1][1] - 1 # Inflow boundary layer of parallel component
    for t in (0.13, 0.71)
        dudt = apply_bc_u(u, t, setup; dudt = true)
        # ∂uin/∂t = π cos(2πt)
        @test all(≈(π * cospi(2t); rtol = 1e-6), view(dudt, i1, Ip.indices[2], 1))
        @test all(iszero, view(dudt, i2, Ip.indices[2], 2))
    end
end

@testitem "Uniform flow with lateral pressure boundaries" begin
    # Uniform flow with constant Dirichlet inflow, and pressure boundaries
    # at the outlet and on both lateral sides (the low side exercises the
    # double-ghost-volume bookkeeping of PressureBC). The flow must be
    # preserved to round-off.
    boundary_conditions =
        (; u = ((DirichletBC((1.0, 0.0)), PressureBC()), (PressureBC(), PressureBC())))
    setup = Setup(; x = (range(0.0, 2.0, 33), range(0.0, 1.0, 17)), boundary_conditions)
    u0 = vectorfield(setup)
    u0[:, :, 1] .= 1.0
    u0 = apply_bc_u(u0, 0.0, setup)
    state, _ = solve_unsteady(;
        setup,
        tlims = (0.0, 0.2),
        start = (; u = u0),
        Δt = 0.005,
        params = (; viscosity = 0.05),
    )
    @test maximum(abs, view(state.u, setup.Iu[1], 1) .- 1) < 1e-11
    @test maximum(abs, view(state.u, setup.Iu[2], 2)) < 1e-11
    @test maximum(abs, view(divergence(state.u, setup), setup.Ip)) < 1e-11
end
