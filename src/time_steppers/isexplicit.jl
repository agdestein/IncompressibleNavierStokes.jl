function isexplicit end

# By default, not explicit
isexplicit(::AbstractODEMethod) = false
isexplicit(::ExplicitRungeKuttaMethod) = true
isexplicit(::OneLegMethod) = true
