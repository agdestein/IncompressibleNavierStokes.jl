"""
    change_time_stepper(stepper, method) -> TimeStepper

Change ODE method. Return a new stepper.
"""
function change_time_stepper end

function change_time_stepper(stepper, method)
    (; setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ) = stepper
    TimeStepper(; method, setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ)
end
