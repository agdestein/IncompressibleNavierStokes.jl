"""
    nstage(rk_stepper)

Get number of stages of the Runge-Kutta time stepper `rk_stepper`.
"""
nstage(rkm::ExplicitRungeKuttaStepper) = length(tableau(rkm)[2])
