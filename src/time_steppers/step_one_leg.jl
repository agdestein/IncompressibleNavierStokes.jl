create_stepper(
    method::OneLegMethod;
    setup,
    psolver,
    u,
    t,
    n = 0,
    p = pressure(u, t, setup; psolver),

    # For the first step, these are not used
    uold = copy.(u),
    pold = copy(p),
    told = t,
) = (; setup, psolver, u, p, t, n, uold, pold, told)

function timestep(method::OneLegMethod, stepper, Δt; θ = nothing)
    (; setup, psolver, u, p, t, n, uold, pold, told) = stepper
    (; p_add_solve, β, method_startup) = method
    (; grid, boundary_conditions) = setup
    T = typeof(Δt)

    # One-leg requires state at previous time step, which is not available at
    # the first iteration. Do one startup step instead
    if n == 0
        stepper_startup = create_stepper(method_startup; setup, psolver, u, t)
        (; u, t, n) = timestep(method_startup, stepper_startup, Δt)
        p = pressure(u, t, setup; psolver)
        return create_stepper(method; setup, psolver, u, p, t, n, uold, pold, told)
    end

    # One-leg requires fixed time step
    @assert Δt ≈ t - told

    # Intermediate ("offstep") velocities
    tβ = t + β * Δt
    uβ = ntuple(α -> @.((1 + β) * u[α] - β * uold[α]), length(u))
    pβ = @. (1 + β) * p - β * pold

    # Right-hand side of the momentum equation
    F = momentum(uβ, tβ, setup)
    G = pressuregradient(pβ, setup)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    unew = ntuple(
        α -> @.(
            (2β * u[α] - (β - T(1) / 2) * uold[α] + Δt * (F[α] - G[α])) / (β + T(1) / 2)
        ),
        length(u),
    )

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary
    # conditions at tₙ₊₁
    unew = apply_bc_u(unew, t + Δt, setup)

    # Adapt time step for pressure calculation
    Δtᵦ = Δt / (β + T(1) / 2)

    # Divergence of intermediate velocity field
    div = divergence(unew, setup) / Δtᵦ
    div = scalewithvolume(div, setup)

    # Solve the Poisson equation for the pressure
    Δp = poisson(psolver, div)
    Δp = apply_bc_p(Δp, t + Δtᵦ, setup)
    GΔp = pressuregradient(Δp, setup)

    # Update velocity field
    unew = ntuple(α -> @.(unew[α] - Δtᵦ * GΔp[α]), length(u))
    unew = apply_bc_u(unew, t + Δt, setup)

    # Update pressure (second order)
    pnew = @. 2 * p - pold + T(4) / 3 * Δp
    pnew = apply_bc_p(pnew, t + Δt, setup)

    # Alternatively, do an additional Poisson solve
    if p_add_solve
        pnew = pressure(unew, t + Δt, setup; psolver)
    end

    told = t
    pold = p
    uold = u
    t = t + Δt
    p = pnew
    u = unew

    create_stepper(method; setup, psolver, u, p, t, n, uold, pold, told)
end

function timestep!(method::OneLegMethod, stepper, Δt; θ = nothing, cache)
    (; setup, psolver, u, p, t, n, uold, pold, told) = stepper
    (; p_add_solve, β, method_startup) = method
    (; grid, boundary_conditions) = setup
    (; unew, pnew, div, F, Δp) = cache
    T = typeof(Δt)

    # One-leg requires state at previous time step, which is not available at
    # the first iteration. Do one startup step instead
    if n == 0
        stepper_startup = create_stepper(method_startup; setup, psolver, u, t)
        (; u, t, n) = timestep(method_startup, stepper_startup, Δt)
        pressure!(p, u, temp, t, setup; psolver, F, div)
        return create_stepper(method; setup, psolver, u, p, t, n, uold, pold, told)
    end

    # One-leg requires fixed time step
    @assert Δt ≈ t - told

    # Intermediate ("offstep") velocities
    tnew = t + β * Δt
    for α = 1:length(u)
        @. unew[α] = (1 + β) * u[α] - β * uold[α]
    end
    @. pnew = (1 + β) * p - β * pold

    # Right-hand side of the momentum equation
    momentum!(F, unew, tnew, setup)
    applypressure!(F, pnew, setup)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    for α = 1:length(u)
        @. unew[α] = 2β * u[α] - (β - T(1) / 2) * uold[α] + Δt * F[α] / (β + T(1) / 2)
    end

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary
    # conditions at tₙ₊₁
    apply_bc_u!(unew, t + Δt, setup)

    # Adapt time step for pressure calculation
    Δtᵦ = Δt / (β + T(1) / 2)

    # Divergence of intermediate velocity field
    divergence!(div, unew, setup)
    scalewithvolume!(div, setup)

    # Solve the Poisson equation for the pressure
    poisson!(psolver, Δp, div)
    apply_bc_p!(Δp, t + Δtᵦ, setup)

    # Update velocity field
    applypressure!(unew, Δp, setup)
    apply_bc_u!(unew, t + Δt, setup)

    # Update pressure (second order)
    @. pnew = 2 * p - pold + T(4) / 3 * Δp / Δtᵦ
    apply_bc_p!(pnew, t + Δt, setup)

    # Alternatively, do an additional Poisson solve
    if p_add_solve
        pressure!(pnew, unew, tempnew, t + Δt, setup; psolver, F, div)
    end

    n += 1
    told = t
    pold .= p
    for α = 1:length(u)
        uold[α] .= u[α]
    end
    t = t + Δt
    p .= pnew
    for α = 1:length(u)
        u[α] .= unew[α]
    end

    create_stepper(method; setup, psolver, u, p, t, n, uold, pold, told)
end
