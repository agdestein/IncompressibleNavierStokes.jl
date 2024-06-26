"""
    rot2(u, r)

Rotate the field `u` by 90 degrees counter-clockwise `r - 1` times.
"""
function rot2 end

# Scalar version
function rot2(u, r)
    nx, ny, s... = size(u)
    @assert nx == ny
    r = mod1(r, 4)
    if r == 1
        i = 1:nx
        j = (1:nx)'
    elseif r == 2
        i = (1:nx)'
        j = nx:-1:1
    elseif r == 3
        i = nx:-1:1
        j = (nx:-1:1)'
    elseif r == 4
        i = (nx:-1:1)'
        j = 1:nx
    end
    I = CartesianIndex.(i, j)
    chans = fill(:, length(s))
    u[I, chans...]
end

# For vector fields (u, v)
function rot2(u::Tuple{T,T}, r) where T
    r = mod1(r, 4)
    ru = rot2(u[1], r)
    rv = rot2(u[2], r)
    if r == 1
        (ru, rv)
    elseif r == 2
        (-rv, ru)
    elseif r == 3
        (-ru, -rv)
    elseif r == 4
        (rv, -ru)
    end
end

# # For augmented vector fields (u, v, -u, -v)
# function rot2(u::Tuple{T,T,T,T}, r) where T
#     nchan = ndims(u[1]) - 2
#     chans = fill(:, nchan)
#     r = mod1(r, 4)
#     ru = rot2.(u, r)
#     if r == 1
#         ru
#     elseif r == 2
#         (-ru[4], ru[1], -ru[2], ru[3])
#     elseif r == 3
#         (-ru[3], -ru[4], -ru[1], -ru[2])
#     elseif r == 4
#         (ru[2], -ru[3], ru[4], -ru[1])
#     end
# end

"""
    GroupConv2D(args...; activation = identity, kwargs...)

Group-equivariant convolutional layer -- with respect to the p4 group.
The layer is equivariant to rotations and translations of the input
vector field.

The `args` and `kwargs` are passed to the `Conv` layer.

If `g = GroupConv2D(...)` is a layer then it should be called on four-dimensional vectors of 2D-coordinates:

```julia
g(u, params, state)
```

where

- `u = (u1, u2, u3, u4)` is a tuple representing the four rotation states
- `u[1]` is a scalar field of size `nx * ny * nchan * nsample` on which
  a normal `Conv` is applied,
- `params` are the `Conv` params,
- `state = (;)` are the empty `Conv` states.
"""
struct GroupConv2D{C,A}
    conv::C
    activation::A
    function GroupConv2D(args...; activation = identity, kwargs...)
        conv = Conv(args...; kwargs...)
        new{typeof(conv),typeof(activation)}(conv, activation)
    end
end

function Lux.initialparameters(rng::AbstractRNG, gc::GroupConv2D)
    params = Lux.initialparameters(rng, gc.conv)
    if haskey(params, :bias)
        params.bias ./= 4
    end
    params
end
Lux.initialstates(rng::AbstractRNG, gc::GroupConv2D) = Lux.initialstates(rng, gc.conv)
Lux.parameterlength(gc::GroupConv2D) = Lux.parameterlength(gc.conv)
Lux.statelength(gc::GroupConv2D) = Lux.statelength(gc.conv)

function (gc::GroupConv2D)(x, params, state)
    (; conv, activation) = gc
    (; weight) = params

    # Compute four new output fields from the four input fields
    # with a "field" being `nx * ny * nchan * nsample` array
    y = ntuple(4) do n
        # Apply convolutions to the four inputs (without any activations)
        y1, _ = conv(x[1], (; params..., weight = rot2(weight, n - 1)), state)
        y2, _ = conv(x[2], (; params..., weight = rot2(weight, n - 2)), state)
        y3, _ = conv(x[3], (; params..., weight = rot2(weight, n - 3)), state)
        y4, _ = conv(x[4], (; params..., weight = rot2(weight, n - 4)), state)

        # Manually apply one activation to the sum.
        # The sum is over the whole p4-group: The four
        # rotation states and all translations.
        # Technically, the same bias appears four times below,
        # but we can pretend btilde = 4b is a new bias.
        @. activation(y1 + y2 + y3 + y4)
    end

    # Return output and unchanged state
    y, state
end
