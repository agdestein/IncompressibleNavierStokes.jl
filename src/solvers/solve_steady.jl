"""
    V, p = solve_steady(setup)

Solve the entire saddlepoint system arising from the steady Navier-Stokes equations with linearization of the convective terms
"""
function solve_steady(setup)
    @unpack Nu, Nv, Np = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack nonlinear_accuracy, Jacobian_type, nPicard, Newton_factor, nonlinear_maxit =
        setup.solversettings
    @unpack use_rom = setup.rom

    Z2 = sparse(Np, Np)

    # Right hand side
    f = zeros(Nu+Nv+Np, 1)

    Newton = false

    maxres[2] = maxres[1]

    while max_residual[n] > nonlinear_accuracy
        if Jacobian_type == "Newton" && nPicard < n
            # Switch to Newton
            setup.solversettings.Newton_factor = true
        end

        n += 1

        _, fmom, dmom = momementum(V, V, p, t, setup, true)
        fmass = M * V + yM
        f = [-fmom; fmass]
        Z = [dfmom -G; -M Z2]
        Δq = Z \ f

        ΔV = Δq[1:Nu+Nv]
        Δp = Δq[Nu+Nv+1:end]

        V = V + ΔV
        p = p + Δp

        ## Process data from this iteration

        # calculate mass, momentum and energy
        maxdiv[n], umom[n], vmom[n], k[n] = check_conservation(V, t, setup)

        # residual (in Finite Volume form)
        # for ke model residual also contains k and e terms and is computed
        # in solver_unsteady_ke
        if use_rom
            # get ROM residual
            maxres[n], _ = F_ROM(R, 0, t, setup, false)
        else
            if visc ≂̸ "keps"
                maxres[n], _ = momentum(V, V, p, t, setup, false)
            end
        end

        # change timestep based on operators
        if !steady && timestep.set && rem(n, timestep.n) == 0
            dt = set_timestep(setup)
        end

        # write restart file
        if restart.write && rem(n, restart.n) == 0
            write_restart()
        end

        # store unsteady data in an array
        if !steady && save_unsteady
            uh_total[n, :] = V[1:setup.grid.Nu]
            vh_total[n, :] = V[setup.grid.Nu+1:end]
            p_total[n, :] = p
        end

        # write convergence information to file
        if setup.output.save_results
            if !steady
                println(
                    fconv,
                    "$n $dt $t $(maxres[n]) $(maxdiv[n]) $(umom[n]) $(vmom[n]) $(k[n])",
                )
            elseif steady
                println(fconv, "$n $(maxres[n]) $(maxdiv[n]) $(umom[n]) $(vmom[n]) $(k[n])")
            end
        end

        println("Residual momentum equation: $(max_residual[n])")

        if n > nonlinear_maxit
            @warn "Newton not converged in $nonlinear_maxit iterations, showing results anyway"
            break
        end
    end

    V, p
end
