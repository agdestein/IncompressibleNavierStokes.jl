"""
    solve_unsteady!(solution, setup)

Main solver file for unsteady calculations
"""
function solve_unsteady!(solution, setup)
    # Setup
    @unpack is_steady, visc = setup.case
    @unpack Nu, Nv, NV, Np, Nx, Ny, x, y = setup.grid
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
        convₙ₋₁ = convection(V, V, t, setup, false)
    end

    # Temporary variables
    cache = MomentumCache(setup)
    F = zeros(NV)
    ∇F = spzeros(NV, NV)

    # for methods that need uₙ₋₁
    Vₙ₋₁ = copy(V)
    pₙ₋₁ = copy(p)

    # Current solution
    Vₙ = copy(V)
    pₙ = copy(p)
    tₙ = t

    # for methods that need extrapolation of convective terms
    if method ∈ [62, 92, 142, 172, 182, 192]
        V_ep = zeros(NV, method_startup_no)
        V_ep[:, 1] .= V
    end

    Δtₙ = Δt

    method_temp = method

    if do_rtp
        ω = get_vorticity(V, t, setup)
        ω = reshape(ω, Nx - 1, Ny - 1)
        ω = Node(ω)
        fig = Figure(resolution = (2000, 300))
        ax, hm = contourf(fig[1, 1], x[2:(end-1)], y[2:(end-1)], ω; levels = -10:2:10)
        ax.aspect = DataAspect()
        ax.title = "Vorticity"
        ax.xlabel = "x"
        ax.ylabel = "y"
        limits!(ax, 0, 10, -0.5, 0.5)
        lines!(ax, [0, 0], [-0.5, 0]; color = :red)
        Colorbar(fig[1, 2], hm)
        display(fig)
        fps = 60
    end

    # record(fig, "output/vorticity.mp4", 2:nt; framerate = 60) do n
    while n ≤ nt
        # Advance one time step
        n += 1
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
            V, p, conv = step_AB_CN!(V, p, Vₙ, pₙ, convₙ₋₁, tₙ, Δt, setup, cache)
            convₙ₋₁ = conv
        elseif method == 5
            step_oneleg!(V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δt, setup, cache)
        elseif method == 20
            step_ERK!(V, p, Vₙ, pₙ, tₙ, Δt, setup, cache, F, ∇F)
        elseif method == 21
            V, p, nonlinear_its[n] = step_IRK(Vₙ, pₙ, tₙ, Δt, setup, cache)
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
            F, = momentum_rom(R, 0, t, setup, false)
            maxres[n] = maximum(abs.(F))
        else
            if visc != "keps"
                # norm of residual
                momentum!(F, ∇F, V, V, p, t, setup, cache, false)
                maxres[n] = maximum(abs.(F))
            end
        end

        # Change timestep based on operators
        if !is_steady && isadaptive && rem(n, n_adapt_Δt) == 0
            Δt = get_timestep(setup)
        end

        # Store unsteady data in an array
        if !is_steady && save_unsteady
            uₕ_total[n, :] = V[indu]
            vₕ_total[n, :] = V[indv]
            p_total[n, :] = p
        end

        if do_rtp #&& mod(n, rtp_n) == 0
            ω[] = reshape(get_vorticity(V, t, setup), Nx - 1, Ny - 1)
            sleep(1 / fps)
        end
    end

    V, p
end
