"""
    bodyforce(force, V, t, setup; get_jacobian = false)

Compute body force `F` in momentum equations at velocity points.

Non-mutating/allocating/out-of-place version.

See also [`bodyforce!`](@ref).
"""
function bodyforce end

bodyforce(force, t, setup) = bodyforce(setup.grid.dimension, force, t, setup)

bodyforce(dimension, force::SteadyBodyForce, t, setup) = force.F

# 2D version
function bodyforce(::Dimension{2}, force::UnsteadyBodyForce, t, setup)
    (; indu, indv, xu, xv, yu, yv) = setup.grid
    Fx = force.fu.(xu, yu, t)
    Fy = force.fv.(xv, yv, t)
    [Fx; Fy]
end

# 3D version
function bodyforce(::Dimension{3}, force::UnsteadyBodyForce, t, setup)
    (; indu, indv, indw, xu, xv, xw, yu, yv, yw, zu, zv, zw) = setup.grid
    Fx = force.fu.(xu, yu, zu, t)
    Fy = force.fv.(xv, yv, zv, t)
    Fz = force.fw.(xw, yw, zw, t)
    [Fx; Fy; Fz]
end

"""
    bodyforce!(force, F, t, setup)

Compute body force `F` in momentum equations at velocity points.

Mutating/non-allocating/in-place version.

See also [`bodyforce`](@ref).
"""
function bodyforce! end

bodyforce!(force, F, t, setup) = bodyforce!(setup.grid.dimension, force, t, F, setup)

bodyforce!(dimension, force::SteadyBodyForce, F, t, setup) = (F .= force.F)

# 2D version
function bodyforce!(::Dimension{2}, force::UnsteadyBodyForce, F, t, setup)
    (; indu, indv, xu, xv, yu, yv) = setup.gri
    F[indu] .= force.fu.(xu, yu, t)
    F[indv] .= force.fv.(xv, yv, t)
    F
end

# 3D version
function bodyforce!(::Dimension{3}, force::UnsteadyBodyForce, F, t, setup)
    (; indu, indv, indw, xu, xv, xw, yu, yv, yw, zu, zv, zw) = setup.grid
    F[indu] .= force.fu.(xu, yu, zu, t)
    F[indv] .= force.fv.(xv, yv, zv, t)
    F[indw] .= force.fw.(xw, yw, zw, t)
    F
end
