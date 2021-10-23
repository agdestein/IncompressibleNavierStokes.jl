function needs_startup_method end

# By deafault, assume that the time stepper does not need a different startup stepper
needs_startup_method(::AbstractODEMethod) = false
needs_startup_method(::OneLegMethod) = true
needs_startup_method(::AdamsBashforthCrankNicolsonMethod) = true
