"""
    $FUNCTIONNAME(method, state, setup)

Get time stepper cache for the given ODE method.

The method `$FUNCTIONNAME(force!, setup)` returns the cache for a
right-hand-side function `force!` instead.
"""
function get_cache end

function get_cache(method::ExplicitRungeKuttaMethod, state, setup)
    ns = length(method.b)
    statestart = map(similar, state)
    k = map(i -> map(similar, state), 1:ns)
    p = scalarfield(setup)
    (; statestart, k, p)
end

function get_cache(::LMWray3, state, setup)
    statestart = map(similar, state)
    k = map(similar, state)
    p = scalarfield(setup)
    (; statestart, k, p)
end
