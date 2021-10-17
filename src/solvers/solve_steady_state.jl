"""
    solve(steady_state_problem, setup, V₀, p₀)

Solve steady state problem of the Navier-Stokes equations.
This saddlepoint system arises from linearization of the convective terms.
"""
function solve(::SteadyStateProblem, setup, V₀, p₀)
    # Setup
    @unpack model = setup
    @unpack problem = setup.case
    @unpack Nu, Nv, NV, Np = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Jacobian_type, nPicard, Newton_factor, nonlinear_acc, nonlinear_maxit =
        setup.solver_settings
    @unpack use_rom = setup.rom
    @unpack do_rtp, rtp_n = setup.visualization
    @unpack t_start = setup.time

    # Temporary variables
    momentum_cache = MomentumCache(setup)
    F = zeros(NV)
    f = zeros(NV + Np)
    ∇F = spzeros(NV, NV)
    Z2 = spzeros(Np, Np)

    # Initialize solution vectors
    # Loop index
    n = 1

    # Initial velocity field
    V = copy(V₀)
    p = copy(p₀)
    t = t_start

    # Initialize BC arrays
    set_bc_vectors!(setup, t)

    # Residual of momentum equations at start
    momentum!(F, ∇F, V, V, p, t, setup, momentum_cache)
    maxres = maximum(abs.(F))

    println("Initial momentum residual = $maxres")

    if do_rtp
        rtp = initialize_rtp(setup, V, p, t)
    end

    # record(fig, "output/vorticity.mp4", 2:rtp.nt; framerate = 60) do n
    while maxres > nonlinear_acc
        if n > nonlinear_maxit
            @warn "Newton not converged in $nonlinear_maxit iterations, showing results anyway"
            break
        end

        print("Iteration $n")

        if Jacobian_type == "newton" && nPicard < n
            # Switch to Newton
            setup.solver_settings.Newton_factor = true
        end

        momentum!(F, ∇F, V, V, p, t, setup, momentum_cache; getJacobian = true)

        fmass = M * V + yM
        f = [-F; fmass]
        Z = [∇F -G; -M Z2]
        Δq = Z \ f

        ΔV = @view Δq[1:NV]
        Δp = @view Δq[NV+1:end]

        V .+= ΔV
        p .+= Δp

        # Calculate mass, momentum and energy
        maxdiv, umom, vmom, k = compute_conservation(V, t, setup)

        if !isa(model, KEpsilonModel)
            momentum!(F, ∇F, V, V, p, t, setup, momentum_cache)
            maxres = maximum(abs.(F))
        end

        println(": momentum residual = $maxres")

        if do_rtp # && mod(n, rtp_n) == 0
            update_rtp!(rtp, setup, V, p, t)
        end

        n += 1
    end

    V, p
end
