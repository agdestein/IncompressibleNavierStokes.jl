"""
    $FUNCTIONNAME(method, setup, u, temp)

Get time stepper cache for the given ODE method.
"""
function get_cache end

function get_cache(::OneLegMethod, setup)
    unew = vectorfield(setup)
    F = vectorfield(setup)
    pnew = scalarfield(setup)
    div = scalarfield(setup)
    Δp = scalarfield(setup)
    (; unew, pnew, F, div, Δp)
end

function get_cache(method::ExplicitRungeKuttaMethod, setup)
    dotemp = !isnothing(setup.temperature)
    ns = length(method.b)
    statestart = (; u = vectorfield(setup), temp = dotemp ? scalarfield(setup) : nothing)
    k = map(
        i -> (; u = vectorfield(setup), temp = dotemp ? scalarfield(setup) : nothing),
        1:ns,
    )
    p = scalarfield(setup)
    (; statestart, k, p)
end

function get_cache(::LMWray3, setup)
    dotemp = !isnothing(setup.temperature)
    statestart = (; u = vectorfield(setup), temp = dotemp ? scalarfield(setup) : nothing)
    k = (; u = vectorfield(setup), temp = dotemp ? scalarfield(setup) : nothing)
    p = scalarfield(setup)
    (; statestart, k, p)
end
