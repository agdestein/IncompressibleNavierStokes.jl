struct VectorConv2D{C,A}
    conv::C
    activation::A
end

function VectorConv2D(args...; activation, kwargs...)
    conv = Conv(args...; kwargs...)
    VectorConv2D{typeof(conv),typeof(activation)}(conv, activation)
end

Lux.initialparameters(rng::AbstractRNG, vc::VectorConv2D) = Lux.initialparameters(rng, vc.conv)
Lux.initialstates(::AbstractRNG, vc::VectorConv2D) = Lux.initialstates(rng, vc.conv)
Lux.parameterlength(vc::VectorConv2D) = Lux.parameterlength(vc.conv)
Lux.statelength(vc::VectorConv2D) = Lux.statelength(vc.conv)

function (vc::VectorConv2D)(xy, params, state)
    (; conv, activation) = vc
    x, y = xy
    cux, _ = conv(x, params, state)
    cuy, _ = conv(y, params, state)
    cvx, _ = conv(x, rotate(rotate(params)), state)
    cvy, _ = conv(y, rotate(params), state)
    u = activation.(cux .+ cuy)
    v = activation.(cvy .+ cvy)
    (u, v), state
end
