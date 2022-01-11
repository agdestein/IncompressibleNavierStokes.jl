"""
    build_force!(force, grid)

Build body force vectors.
"""
function build_force! end

# 2D version
function build_force!(force::SteadyBodyForce{T}, grid::Grid{T,2}) where T
    (; bodyforce_u, bodyforce_v) = force
    (; NV, indu, indv, xu, yu, xv, yv) = grid

    F = zeros(T, NV) 
    Fu = @view F[indu]
    Fv = @view F[indv]
    Fu .= reshape(bodyforce_u.(xu, yu), :)
    Fv .= reshape(bodyforce_v.(xv, yv), :)

    @pack! force = F

    force
end

# 3D version
function build_force!(force::SteadyBodyForce{T}, grid::Grid{T,3}) where T
    (; bodyforce_u, bodyforce_v, bodyforce_w) = force
    (; NV, indu, indv, indw, xu, yu, zu, xv, yv, zv, xw, yw, zw) = grid

    F = zeros(T, NV) 
    Fu = @view F[indu]
    Fv = @view F[indv]
    Fw = @view F[indw]
    Fu .= reshape(bodyforce_u.(xu, yu, zu), :)
    Fv .= reshape(bodyforce_v.(xv, yv, zv), :)
    Fw .= reshape(bodyforce_w.(xw, yw, zw), :)

    @pack! force = F

    force
end

function build_force!(force::UnsteadyBodyForce{T}, grid) where T
    (; bodyforce_u, bodyforce_v) = force
    (; NV, indu, indv, xu, yu, xv, yv) = grid

    error("Not implemented")

    force
end

