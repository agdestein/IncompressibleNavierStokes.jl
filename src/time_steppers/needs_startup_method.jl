"""
    needs_startup_method(method)

Return `true` if `method` needs a startup method to generate an initial
history.
"""
function needs_startup_method end

# By default, assume that the time stepper does not need a different startup
# stepper
needs_startup_method(::AbstractODEMethod) = false
needs_startup_method(::OneLegMethod) = true
needs_startup_method(::AdamsBashforthCrankNicolsonMethod) = true
