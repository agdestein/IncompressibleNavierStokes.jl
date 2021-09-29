"""
    solve_unsteady!(solution, setup)

Main solver file for unsteady calculations
"""
function solve_unsteady!(solution, setup)
    # Setup
    @unpack is_steady, visc = setup.case
    @unpack Nu, Nv, Np, Nx, Ny, x, y = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Jacobian_type, nPicard, Newton_factor, nonlinear_acc, nonlinear_maxit =
        setup.solver_settings
    @unpack use_rom = setup.rom
    @unpack isadaptive, method, method_startup = setup.time
    @unpack save_unsteady = setup.output
    @unpack do_rtp, rtp_n = setup.visualization

    # Solution
    @unpack V, p, t, Δt, n, nt, maxres, maxdiv, k, vmom, nonlinear_its, time, umom =
        solution

    # for methods that need convection from previous time step
    if method == 2
        if setup.bc.bc_unsteady
            set_bc_vectors!(setup, t)
        end
        convuₙ₋₁, convvₙ₋₁ = convection(V, V, t, setup, false)
        convₙ₋₁ = [convuₙ₋₁; convvₙ₋₁]
    end

    # for methods that need uₙ₋₁
    Vₙ₋₁ = copy(V)
    pₙ₋₁ = copy(p)

    # Current solution
    Vₙ = copy(V)
    pₙ = copy(p)
    tₙ = t

    # for methods that need extrapolation of convective terms
    if method ∈ [62, 92, 142, 172, 182, 192]
        V_ep = zeros(Nu + Nv, method_startup_no)
        V_ep[:, 1] .= V
    end

    Δtₙ = Δt

    method_temp = method

    if do_rtp
        ω = get_vorticity(V, t, setup)
        ω = reshape(ω, Nx - 1, Ny - 1)
        ω = Node(ω)
        pl = contourf(x[2:(end-1)], y[2:(end-1)], ω)
        display(pl)
    end

    while n ≤ nt
        # Advance one time step
        n = n + 1
        Vₙ₋₁ .= Vₙ
        pₙ₋₁ .= pₙ
        Vₙ .= V
        pₙ .= p
        tₙ = t

        # for methods that need a velocity field at n-1 the first time step
        # (e.g. AB-CN, oneleg beta) use ERK or IRK
        if method_temp ∈ [2, 5] && n ≤ method_startup_no
            println("Starting up with method $method_startup")
            method = method_startup
        else
            method = method_temp
        end

        # Perform a single time step with the time integration method
        if method == 2
            V, p, conv = step_AB_CN!(V, p, Vₙ, pₙ, convₙ₋₁, tₙ, Δt, setup)
            convₙ₋₁ = conv
        elseif method == 5
            step_oneleg!(V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δt, setup)
        elseif method == 20
            step_ERK!(V, p, Vₙ, pₙ, tₙ, Δt, setup)
        elseif method == 21
            V, p, nonlinear_its[n] = step_IRK(Vₙ, pₙ, tₙ, Δt, setup)
        else
            error("time integration method unknown")
        end

        # the velocities and pressure that are just computed are at
        # the new time level t+Δt:
        t = tₙ + Δt
        time[n] = t

        ## Process data from this iteration
        # check residuals, conservation, set timestep, write output files

        # calculate mass, momentum and energy
        maxdiv[n], umom[n], vmom[n], k[n] = check_conservation(V, t, setup)

        # residual (in Finite Volume form)
        # for ke model residual also contains k and e terms and is computed
        # in solver_unsteady_ke
        if use_rom
            # get ROM residual
            maxres[n], = F_ROM(R, 0, t, setup, false)
        else
            if visc != "keps"
                maxres[n], = momentum(V, V, p, t, setup, false)
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

        if do_rtp && mod(n, rtp_n) == 0
            ω[] = reshape(get_vorticity(V, t, setup), Nx - 1, Ny - 1)
            # display(pl)
            sleep(0.05)
        end
    end

    V, p
end
