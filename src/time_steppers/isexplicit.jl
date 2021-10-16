function isexplicit end

# By default, not explicit
isexplicit(::TimeStepper) = false
isexplicit(::ExplicitRungeKuttaStepper) = true
isexplicit(::OneLegStepper) = true
