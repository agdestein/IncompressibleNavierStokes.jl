struct VectorConv2D{C,A}
    conv::C
    activation::A
    function VectorConv2D(args...; activation, kwargs...)
        conv = Conv(args...; kwargs...)
        new{typeof(conv),typeof(activation)}(conv, activation)
    end
end

Lux.initialparameters(rng::AbstractRNG, vc::VectorConv2D) =
    Lux.initialparameters(rng, vc.conv)
Lux.initialstates(::AbstractRNG, vc::VectorConv2D) = Lux.initialstates(rng, vc.conv)
Lux.parameterlength(vc::VectorConv2D) = Lux.parameterlength(vc.conv)
Lux.statelength(vc::VectorConv2D) = Lux.statelength(vc.conv)

function rotate(w)
    s = size(w, 1)
    i = 1:s
    w[i, j, ch]
end

function (vc::VectorConv2D)(xy, params, state)
    (; conv, activation) = vc
    (; weight) = params
    x, y = xy
    wux = weight
    wuy = rotate(weight)
    wvx = rotate(rotate(weight))
    wvy = rotate(weight)
    cux, _ = conv(x, (; params..., weight = wux), state)
    cuy, _ = conv(y, (; params..., weight = wuy), state)
    cvx, _ = conv(x, (; params..., weight = wvx), state)
    cvy, _ = conv(y, (; params..., weight = wvy), state)
    u = activation.(cux .+ cuy)
    v = activation.(cvx .+ cvy)
    (u, v), state
end
