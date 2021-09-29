"""
    solve_steady!(solution, setup)

Solve the entire saddlepoint system arising from the steady Navier-Stokes equations with linearization of the convective terms
"""
function solve_steady!(solution, setup)
    # Setup
    @unpack is_steady, visc = setup.case
    @unpack Nu, Nv, Np = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Jacobian_type, nPicard, Newton_factor, nonlinear_acc, nonlinear_maxit =
        setup.solver_settings
    @unpack use_rom = setup.rom

    # Solution
    @unpack V, p, t, Δt, n, maxres, maxdiv, k, vmom, nonlinear_its, time, umom = solution

    Z2 = spzeros(Np, Np)

    # Right hand side
    f = zeros(Nu+Nv+Np)

    Newton = false

    maxres[2] = maxres[1]

    while maxres[n] > nonlinear_acc
        if Jacobian_type == "Newton" && nPicard < n
            # Switch to Newton
            setup.solver_settings.Newton_factor = true
        end

        n += 1

        fmom, dfmom = momentum(V, V, p, t, setup, true)
        fmass = M * V + yM
        f = [-fmom; fmass]
        Z = [dfmom -G; -M Z2]
        Δq = Z \ f

        ΔV = @view Δq[1:Nu+Nv]
        Δp = @view Δq[Nu+Nv+1:end]

        V .+= ΔV
        p .+= Δp

        ## Process data from this iteration

        # calculate mass, momentum and energy
        maxdiv[n], umom[n], vmom[n], k[n] = check_conservation(V, t, setup)

        # residual (in Finite Volume form)
        # for ke model residual also contains k and e terms and is computed
        # in solver_unsteady_ke
        if use_rom
            # get ROM residual
            Fres = momentum_rom(R, 0, t, setup, false)
            maxres[n] = maximum(abs.(Fres))
        else
            if visc != "keps"
                Fres, = momentum(V, V, p, t, setup, false)
                maxres[n] = maximum(abs.(Fres))
            end
        end

        # change timestep based on operators
        if !is_steady && isadaptive && rem(n, n_adapt_Δt) == 0
            Δt = get_timestep(setup)
        end

        # store unsteady data in an array
        if !is_steady && save_unsteady
            uₕ_total[n, :] = V[1:setup.grid.Nu]
            vₕ_total[n, :] = V[setup.grid.Nu+1:end]
            p_total[n, :] = p
        end

        # write convergence information to file
        if setup.output.save_results
            if !is_steady
                println(
                    fconv,
                    "$n $Δt $t $(maxres[n]) $(maxdiv[n]) $(umom[n]) $(vmom[n]) $(k[n])",
                )
            elseif is_steady
                println(fconv, "$n $(maxres[n]) $(maxdiv[n]) $(umom[n]) $(vmom[n]) $(k[n])")
            end
        end

        println("Iteration $n: momentum residual = $(maxres[n])")

        if n > nonlinear_maxit
            @warn "Newton not converged in $nonlinear_maxit iterations, showing results anyway"
            break
        end
    end

    V, p
end
