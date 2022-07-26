"""
    bodyforce(V, t, setup; getJacobian = false)

Body force in momentum equations in Finite Volume setting, so integrated `dFx`, `dFy` are
the Jacobians `∂Fx/∂V` and `∂Fy/∂V`.
"""
function bodyforce end

bodyforce(force::SteadyBodyForce, t, setup) = force.F

# 2D version
function bodyforce(force::UnsteadyBodyForce, t, setup::Setup{T,2}) where {T}
    (; indu, indv, xu, xv, yu, yv) = setup.grid
    Fx = setup.force.bodyforce_x.(xu, yu, t)
    Fy = setup.force.bodyforce_y.(xv, yv, t)
    [Fx; Fy]
end

# 3D version
function bodyforce(force::UnsteadyBodyForce, t, setup::Setup{T,3}) where {T}
    (; indu, indv, indw, xu, xv, xw, yu, yv, yw, zu, zv, zw) = setup.grid
    Fx = setup.force.bodyforce_x.(xu, yu, zu, t)
    Fy = setup.force.bodyforce_y.(xv, yv, zv, t)
    Fz = setup.force.bodyforce_z.(xw, yw, zw, t)
    [Fx; Fy; Fz]
end


"""
    bodyforce!(F, t, setup)

Compute body force `F` in momentum equations at velocity points.
If `getJacobian`, also compute `∇F = ∂F/∂V`.
"""
function bodyforce! end

bodyforce!(force::SteadyBodyForce, F, t, setup) = (F .= force.F)

# 2D version
function bodyforce!(force::UnsteadyBodyForce, F, t, setup::Setup{T,2}) where {T}
    (; indu, indv, xu, xv, yu, yv) = setup.grid
    F[indu] .= force.bodyforce_x.(xu, yu, t)
    F[indv] .= force.bodyforce_y.(xv, yv, t)
    F
end

# 3D version
function bodyforce!(force::UnsteadyBodyForce, F, t, setup::Setup{T,3}) where {T}
    (; indu, indv, indw, xu, xv, xw, yu, yv, yw, zu, zv, zw) = setup.grid
    F[indu] .= force.bodyforce_x.(xu, yu, zu, t)
    F[indv] .= force.bodyforce_y.(xv, yv, zv, t)
    F[indw] .= force.bodyforce_z.(xw, yw, zw, t)
    F
end

