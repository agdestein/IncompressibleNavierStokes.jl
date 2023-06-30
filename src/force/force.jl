"""
    SteadyBodyForce(fu, fv, grid)

Two-dimensional steady body force `f(x, y) = [fu(x, y), fv(x, y)]`. 
"""
function SteadyBodyForce(fu, fv, grid)
    (; NV, indu, indv, xu, yu, xv, yv) = grid
    T = eltype(xu)
    F = zeros(T, NV)
    F[indu] .= reshape(fu.(xu, yu), :)
    F[indv] .= reshape(fv.(xv, yv), :)
    F
end

"""
    SteadyBodyForce(fu, fv, fw, grid)

Three-dimensional steady body force `f(x, y, z) = [fu(x, y, z), fv(x, y, z), fw(x, y, z)]`. 
"""
function SteadyBodyForce(fu, fv, fw, grid)
    (; NV, indu, indv, indw, xu, yu, zu, xv, yv, zv, xw, yw, zw) = grid
    T = eltype(xu)
    F = zeros(T, NV)
    F[indu] .= reshape(fu.(xu, yu, zu), :)
    F[indv] .= reshape(fv.(xv, yv, zv), :)
    F[indw] .= reshape(fw.(xw, yw, zw), :)
    F
end
