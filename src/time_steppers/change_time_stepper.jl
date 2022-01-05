"""
    change_time_stepper(stepper, method) -> AbstractTimeStepper

Change ODE method. Return a new stepper.
"""
function change_time_stepper end

function change_time_stepper(stepper, method)
    (; n, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup) = stepper
    new_stepper = TimeStepper(method, setup, V, p)
    @pack! new_stepper = n, V, p, t, Vₙ, pₙ, tₙ, Δtₙ
    new_stepper
end
