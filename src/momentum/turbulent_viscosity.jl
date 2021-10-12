"""
    turbulent_viscosity(S_abs, setup)

Compute turbulent viscosity based on `S_abs`.
"""
function turbulent_viscosity(S_abs, setup)
    visc = setup.case.visc
    if visc == "LES"
        # Smagorinsky
        C_S = setup.visc.Cs
        filter_length = deltax # = sqrt(FV size) for uniform grids
        nu_t = C_S^2 * filter_length^2 * S_abs
    elseif visc == "qr"
        # Q-r
        C_d = deltax^2 / 8
        nu_t = C_d * 0.5 * S_abs * (1 - α / C_d)^2
    elseif visc == "ML"
        # Mixing-length
        lm = setup.visc.lm # Mixing length
        nu_t = (lm^2) * S_abs
    else
        error("Wrong value for visc parameter")
    end
    nu_t
end
