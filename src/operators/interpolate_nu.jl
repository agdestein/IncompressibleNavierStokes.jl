"""
    ν_t_ux, ν_t_uy, ν_t_vx, ν_t_vy = interpolate_ν(ν_t, setup)
interpolate the scalar field ν_t at pressure locations (xp, yp)
to locations needed in computing the diffusive terms, i.e. the u_x, u_y,
v_x and v_y locations
"""
function interpolate_ν(ν_t, setup)
    (; Aν_ux, Aν_uy, Aν_vx, Aν_vy) = setup.operators
    ν_t_ux = Aν_ux * ν_t
    ν_t_uy = Aν_uy * ν_t
    ν_t_vx = Aν_vx * ν_t
    ν_t_vy = Aν_vy * ν_t
    ν_t_ux, ν_t_uy, ν_t_vx, ν_t_vy
end
