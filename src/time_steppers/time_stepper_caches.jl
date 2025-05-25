"""
    $FUNCTIONNAME(method, setup, u, temp)

Get time stepper cache for the given ODE method.
"""
function get_cache end

function get_cache(::OneLegMethod, state, setup)
    # TODO
    unew = vectorfield(setup)
    F = vectorfield(setup)
    pnew = scalarfield(setup)
    div = scalarfield(setup)
    Δp = scalarfield(setup)
    (; unew, pnew, F, div, Δp)
end

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
