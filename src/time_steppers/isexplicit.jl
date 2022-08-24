"""
    isexplicit(method)

Return `true` if `method` is explicit, i.e. the value at a certain time step is
given explicitly as a function of the previous time steps onl is given
explicitly as a function of the previous time steps only.
"""
function isexplicit end

# By default, not explicit
isexplicit(::AbstractODEMethod) = false
isexplicit(::ExplicitRungeKuttaMethod) = true
isexplicit(::OneLegMethod) = true
