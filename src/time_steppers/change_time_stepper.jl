"""
    change_time_stepper(stepper, method) -> AbstractTimeStepper

Change ODE method. Return a new stepper.
"""
function change_time_stepper end

function change_time_stepper(stepper, method)
    @unpack n, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup = stepper
    new = TimeStepper(method, setup, V, p) 
    @pack! new = n, V, p, t, Vₙ, pₙ, tₙ, Δtₙ
    new
end
