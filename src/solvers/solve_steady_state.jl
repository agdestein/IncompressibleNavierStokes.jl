"""
    function solve(problem::SteadyStateProblem; processors = Processor[])

Solve steady state problem of the Navier-Stokes equations.
This saddlepoint system arises from linearization of the convective terms.

Each `processor` is called after every `processor.nupdate` iteration.
"""
function solve(problem::SteadyStateProblem; processors = Processor[])
    (; setup, V₀, p₀) = problem
    (; NV, Np) = setup.grid
    (; G, M, yM) = setup.operators
    (; Jacobian_type, nPicard, nonlinear_acc, nonlinear_maxit) = setup.solver_settings

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
    t = 0.0

    # Initialize BC arrays
    set_bc_vectors!(setup, t)

    # Residual of momentum equations at start
    momentum!(F, ∇F, V, V, p, t, setup, momentum_cache)
    maxres = maximum(abs.(F))

    println("Initial momentum residual = $maxres")

    # Processors for iteration results  
    for ps ∈ processors
        # initialize!(ps, stepper)
    end

    # record(fig, "output/vorticity.mp4", 1:rtp.nt; framerate = 60) do n
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

        momentum!(F, ∇F, V, V, p, t, setup, momentum_cache)
        maxres = maximum(abs.(F))

        println(": momentum residual = $maxres")

        # Process iteration results with each processor
        for ps ∈ processors
            # Only update each `nupdate`-th iteration
            # stepper.n % ps.nupdate == 0 && process!(ps, stepper)
        end

        # finalize!.(processors)

        n += 1
    end

    V, p
end
