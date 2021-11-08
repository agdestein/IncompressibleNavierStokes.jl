"""
    build_force!(force, grid)

Build body force vectors.
"""
function build_force! end

function build_force!(force::SteadyBodyForce{T}, grid) where T
    @unpack bodyforce_u, bodyforce_v = force
    @unpack NV, indu, indv, xu, yu, xv, yv = grid

    F = zeros(T, NV) 
    Fu = @view F[indu]
    Fv = @view F[indv]
    Fu .= reshape(bodyforce_u.(xu, yu), :)
    Fv .= reshape(bodyforce_v.(xv, yv), :)

    @pack! force = F

    force
end

function build_force!(force::UnsteadyBodyForce{T}, grid) where T
    @unpack bodyforce_u, bodyforce_v = force
    @unpack NV, indu, indv, xu, yu, xv, yv = grid

    error("Not implemented")

    force
end

