"""
    $FUNCTIONNAME(method, setup, u, temp)

Get time stepper cache for the given ODE method.
"""
function ode_method_cache end

function ode_method_cache(::OneLegMethod, setup)
    unew = vectorfield(setup)
    F = vectorfield(setup)
    pnew = scalarfield(setup)
    div = scalarfield(setup)
    Δp = scalarfield(setup)
    (; unew, pnew, F, div, Δp)
end

function ode_method_cache(method::ExplicitRungeKuttaMethod, setup)
    ustart = vectorfield(setup)
    ns = length(method.b)
    ku = map(i -> vectorfield(setup), 1:ns)
    p = scalarfield(setup)
    if isnothing(setup.temperature)
        tempstart = nothing
        ktemp = nothing
        diff = nothing
    else
        tempstart = scalarfield(setup)
        ktemp = map(i -> scalarfield(setup), 1:ns)
        diff = vectorfield(setup)
    end
    (; ustart, ku, p, tempstart, ktemp, diff)
end

function ode_method_cache(::LMWray3, setup)
    ustart = vectorfield(setup)
    ku = vectorfield(setup)
    p = scalarfield(setup)
    if isnothing(setup.temperature)
        tempstart = nothing
        ktemp = nothing
        diff = nothing
    else
        tempstart = scalarfield(setup)
        ktemp = scalarfield(setup)
        diff = vectorfield(setup)
    end
    (; ustart, ku, p, tempstart, ktemp, diff)
end
