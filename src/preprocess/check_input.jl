"""
Check input.
"""
function check_input!(setup, V₀, p₀, t₀)
    @unpack model = setup
    @unpack problem = setup.case
    @unpack order4 = setup.discretization
    @unpack t_start, t_end, Δt, isadaptive, method, method_startup = setup.time

    if order4
        if !isa(model, LaminarModel)
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
    if is_steady(problem)
        setup.solver_settings.Newton_factor = false
    elseif method isa ImplicitRungeKuttaMethod || method_startup isa ImplicitRungeKuttaMethod
        # Implicit RK time integration
        setup.solver_settings.Newton_factor = true
    end
end
