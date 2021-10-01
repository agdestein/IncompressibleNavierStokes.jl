"""
    solve_steady!(solution, setup)

Solve the entire saddlepoint system arising from the steady Navier-Stokes equations with linearization of the convective terms
"""
function solve_steady!(solution, setup)
    # Setup
    @unpack is_steady, visc = setup.case
    @unpack Nu, Nv, NV, Np = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Jacobian_type, nPicard, Newton_factor, nonlinear_acc, nonlinear_maxit =
        setup.solver_settings
    @unpack use_rom = setup.rom

    # Solution
    @unpack V, p, t, Δt, n, maxres, maxdiv, k, vmom, nonlinear_its, time, umom = solution

    # Temporary variables
    cache = MomentumCache(setup)
    F = zeros(NV)
    f = zeros(NV + Np)
    ∇F = spzeros(NV, NV)
    Z2 = spzeros(Np, Np)

    maxres[2] = maxres[1]

    while maxres[n] > nonlinear_acc
        if Jacobian_type == "newton" && nPicard < n
            # Switch to Newton
            setup.solver_settings.Newton_factor = true
        end

        n += 1

        momentum!(F, ∇F, V, V, p, t, setup, cache, true)

        fmass = M * V + yM
        f = [-F; fmass]
        Z = [∇F -G; -M Z2]
        Δq = Z \ f

        ΔV = @view Δq[1:NV]
        Δp = @view Δq[NV+1:end]

        V .+= ΔV
        p .+= Δp

        # Calculate mass, momentum and energy
        maxdiv[n], umom[n], vmom[n], k[n] = check_conservation(V, t, setup)

        if visc != "keps"
            momentum!(F, ∇F, V, V, p, t, setup, cache, false)
            maxres[n] = maximum(abs.(F))
        end

        println("Iteration $n: momentum residual = $(maxres[n])")

        if n > nonlinear_maxit
            @warn "Newton not converged in $nonlinear_maxit iterations, showing results anyway"
            break
        end
    end

    V, p
end
