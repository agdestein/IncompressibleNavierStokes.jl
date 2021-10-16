function needs_startup_stepper end

# By deafault, assume that the time stepper does not need a different startup stepper
needs_startup_stepper(::TimeStepper) = false
needs_startup_stepper(::OneLegStepper) = true
needs_startup_stepper(::AdamsBashforthCrankNicolsonStepper) = true
