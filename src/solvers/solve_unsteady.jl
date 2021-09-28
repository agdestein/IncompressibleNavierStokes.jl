"""
    solve_unsteady!(solution, setup)

Main solver file for unsteady calculations
"""
function solve_unsteady!(solution, setup)
    # Setup
    @unpack steady, visc = setup.case
    @unpack Nu, Nv, Np = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Jacobian_type, nPicard, Newton_factor, nonlinear_acc, nonlinear_maxit =
        setup.solver_settings
    @unpack use_rom = setup.rom
    @unpack isadaptive, method, method_startup = setup.time
    @unpack save_unsteady = setup.output

    # Solution
    @unpack V, p, t, Δt, n, nt, maxres, maxdiv, k, vmom, nonlinear_its, time, umom = solution

    # for methods that need convection from previous time step
    if method == 2
        if setup.bc.bc_unsteady
            set_bc_vectors(t, setup)
        end
        convu_old, convv_old = convection(V, V, t, setup, false)
        conv_old = [convu_old; convv_old]
    end

    # for methods that need u^(n-1)
    if method == 5
        V_old = copy(V)
        p_old = copy(p)
    end

    # set current velocity and pressure
    Vₙ = V
    pₙ = p
    tₙ = t

    # for methods that need extrapolation of convective terms
    if method ∈ [62, 92, 142, 172, 182, 192]
        V_ep = zeros(Nu + Nv, method_startup_no)
        V_ep[:, 1] .= V
    end

    Δtₙ = Δt

    method_temp = method

    while n <= nt
        # time step counter
        n = n + 1

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
            V, p, conv = time_AB_CN(Vₙ, pₙ, conv_old, tₙ, Δt, setup)
            conv_old = conv
        elseif method == 5
            V, p = step_oneleg(Vₙ, pₙ, V_old, p_old, tₙ, Δt, setup)
        elseif method == 20
            V, p = step_ERK(Vₙ, pₙ, tₙ, Δt, setup)
        elseif method == 21
            V, p, nonlinear_its[n] = step_IRK(Vₙ, pₙ, tₙ, Δt, setup)
        else
            error("time integration method unknown")
        end

        # the velocities and pressure that are just computed are at
        # the new time level t+Δt:
        t = tₙ + Δt
        time[n] = t

        # Update solution
        V_old = Vₙ
        p_old = pₙ
        Vₙ = V
        pₙ = p
        tₙ = t

        ## Process data from this iteration
        # check residuals, conservation, set timestep, write output files

        # calculate mass, momentum and energy
        maxdiv[n], umom[n], vmom[n], k[n] = check_conservation(V, t, setup)

        # residual (in Finite Volume form)
        # for ke model residual also contains k and e terms and is computed
        # in solver_unsteady_ke
        if use_rom
            # get ROM residual
            maxres[n], _ = F_ROM(R, 0, t, setup, false)
        else
            if visc != "keps"
                maxres[n], _ = momentum(V, V, p, t, setup, false)
            end
        end

        # change timestep based on operators
        if !steady && isadaptive && rem(n, n_adapt_Δt) == 0
            Δt = get_timestep(setup)
        end

        # store unsteady data in an array
        if !steady && save_unsteady
            uₕ_total[n, :] = V[1:setup.grid.Nu]
            vₕ_total[n, :] = V[setup.grid.Nu+1:end]
            p_total[n, :] = p
        end
    end

    V, p
end
