# 3D version
function plot_force(setup::Setup{T,3}, F, t; kwargs...) where {T}
    (; xu, yu, zu, indu) = setup.grid

    # Get force at pressure points
    # up, vp, wp = get_velocity(V, t, setup)
    # qp = map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
    Fp = reshape(F[indu], size(xu))
    xp, yp, zp = xu[:,1,1], yu[1,:,1], zu[1,1,:]
    
    # Levels
    μ, σ = mean(Fp), std(Fp)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)

    # contour(xp, yp, zp, Fp; levels, kwargs...)
    contour(xp, yp, zp, Fp)
end
