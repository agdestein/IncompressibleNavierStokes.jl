"""
    nstage(rk_method)

Get number of stages of the Runge-Kutta method.
"""
nstage(rkm::AbstractRungeKuttaMethod) = length(rkm.b)
