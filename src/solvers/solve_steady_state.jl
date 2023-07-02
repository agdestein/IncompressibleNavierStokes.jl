"""
    function solve_steady_state(
        setup, V₀, p₀;
        jacobian_type = :newton,
        npicard = 2,
        abstol = 1e-10,
        maxiter = 10,
    )

Solve steady state problem of the Navier-Stokes equations.
This saddlepoint system arises from linearization of the convective terms.

Each `processor` is called after every `processor.nupdate` iteration.
"""
function solve_steady_state(
    setup, V₀, p₀;
    jacobian_type = :newton,
    npicard = 2,
    abstol = 1e-10,
    maxiter = 10,
)
    (; NV, Np) = setup.grid
    (; G, M) = setup.operators

    T = eltype(V₀)

    # Temporary variables
    momentum_cache = MomentumCache(setup)
    F = zeros(T, NV)
    f = zeros(T, NV + Np)
    ∇F = spzeros(T, NV, NV)
    Z2 = spzeros(T, Np, Np)
    Δq = zeros(T, NV + Np)

    # Loop index
    n = 1

    # Initial velocity field
    V = copy(V₀)
    p = copy(p₀)
    t = 0.0

    # Start with Picard iterations
    newton_factor = false

    # Initialize BC arrays
    bc_vectors = get_bc_vectors(setup, t)
    (; yM) = bc_vectors

    # Residual of momentum equations at start
    momentum!(F, ∇F, V, V, p, t, setup, momentum_cache; bc_vectors)
    maxres = maximum(abs.(F))

    println("Initial momentum residual = $maxres")

    # record(fig, "output/vorticity.mp4", 1:rtp.nt; framerate = 60) do n
    while maxres > abstol
        if n > maxiter
            @warn "Newton not converged in $maxiter iterations, showing results anyway"
            break
        end

        print("Iteration $n")

        if jacobian_type == :newton && npicard < n
            # Switch to Newton
            newton_factor = true
        end

        momentum!(
            F,
            ∇F,
            V,
            V,
            p,
            t,
            setup,
            momentum_cache;
            bc_vectors,
            get_jacobian = true,
            newton_factor,
        )

        fmass = M * V + yM
        f = [-F; fmass]
        Z = [∇F -G; -M Z2]

        Δq = Z \ f
        # gmres!(Δq, Z, f)
        # bicgstabl!(Δq, Z, f)

        ΔV = @view Δq[1:NV]
        Δp = @view Δq[(NV+1):end]

        V .+= ΔV
        p .+= Δp

        # Calculate mass, momentum and energy
        # maxdiv, umom, vmom, k = compute_conservation(V, t, setup; bc_vectors)

        momentum!(F, ∇F, V, V, p, t, setup, momentum_cache; bc_vectors)
        maxres = maximum(abs.(F))

        println(": momentum residual = $maxres")

        n += 1
    end

    V, p
end
