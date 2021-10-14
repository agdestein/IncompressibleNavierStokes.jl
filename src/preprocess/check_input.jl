"""
Check input.
"""
function check_input!(setup, V₀, p₀, t₀)
    @unpack is_steady, visc = setup.case
    @unpack order4 = setup.discretization
    @unpack t_start, t_end, Δt, isadaptive, method, method_startup = setup.time
    @unpack save_unsteady = setup.output

    if order4
        if visc != "laminar"
            error(
                "order 4 only implemented for laminar flow; need to add Su_vx and Sv_uy for 4th order method",
            )
        end

        if regularization != "no"
            error(
                "order 4 only implemented for standard convection with regularization == \"no\"",
            )
        end
    end

    symmetry_flag, symmetry_error = check_symmetry(V₀, t₀, setup)

    # For steady problems, with Newton linearization and full Jacobian, first start with nPicard Picard iterations
    if setup.case.is_steady
        setup.solver_settings.Newton_factor = false
    elseif method == 21 || method_startup == 21
        # Implicit RK time integration
        setup.solver_settings.Newton_factor = true
    end
end
