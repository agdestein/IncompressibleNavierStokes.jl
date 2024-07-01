function rot2mat(n)
    n = mod(n, 4)
    if n == 0
        [1 0; 0 1]
    elseif n == 1
        [0 -1; 1 0]
    elseif n == 2
        [-1 0; 0 -1]
    elseif n == 3
        [0 1; -1 0]
    end
end

"""
    rot2(u, r)

Rotate the field `u` by 90 degrees counter-clockwise `r - 1` times.
"""
function rot2 end

# Scalar version
function rot2(u, r)
    nx, ny, s... = size(u)
    @assert nx == ny
    r = mod(r, 4)
    if r == 0
        i = 1:nx
        j = (1:nx)'
    elseif r == 1
        i = (nx:-1:1)'
        j = 1:nx
    elseif r == 2
        i = nx:-1:1
        j = (nx:-1:1)'
    elseif r == 3
        i = (1:nx)'
        j = nx:-1:1
    end
    I = CartesianIndex.(i, j)
    chans = fill(:, length(s))
    u[I, chans...]
end

# For vector fields (u, v)
function rot2(u::Tuple{T,T}, r) where {T}
    r = mod(r, 4)
    ru = rot2(u[1], r)
    rv = rot2(u[2], r)
    if r == 0
        (ru, rv)
    elseif r == 1
        (-rv, ru)
    elseif r == 2
        (-ru, -rv)
    elseif r == 3
        (rv, -ru)
    end
end

# # For augmented vector fields (u, v, -u, -v)
# function rot2(u::Tuple{T,T,T,T}, r) where T
#     nchan = ndims(u[1]) - 2
#     chans = fill(:, nchan)
#     r = mod(r, 4)
#     ru = rot2.(u, r)
#     if r == 0
#         ru
#     elseif r == 1
#         (-ru[4], ru[1], -ru[2], ru[3])
#     elseif r == 2
#         (-ru[3], -ru[4], -ru[1], -ru[2])
#     elseif r == 3
#         (ru[2], -ru[3], ru[4], -ru[1])
#     end
# end

"""
    GroupConv2D(
        k,
        chans,
        activation = identity;
        islifting = false,
        isprojecting = false,
        kwargs...,
    )

Group-equivariant convolutional layer -- with respect to the p4 group.
The layer is equivariant to rotations and translations of the input
vector field.

The `kwargs` are passed to the `Conv` layer.

The layer has three variants:

- If `islifting` then it lifts a vector input `(u1, u2)` into a rotation-state vector `(v1, v2, v3, v4)`.
- If `isprojecting`, it projects a rotation-state vector `(u1, u2, u3, v4)` into a vector `(v1, v2)`.
- Otherwise, it cyclically transforms the rotation-state vector `(u1, u2, u3, u4)` into a new rotation-state vector `(v1, v2, v3, v4)`.
"""
struct GroupConv2D{C} <: Lux.AbstractExplicitLayer
    islifting::Bool
    isprojecting::Bool
    cin::Int
    cout::Int
    conv::C
    function GroupConv2D(
        k,
        chans,
        activation = identity;
        islifting = false,
        isprojecting = false,
        kwargs...,
    )
        @assert !(islifting && isprojecting) "Cannot lift and project"

        # New channel size: Two velocity fields and four rotation states
        cin, cout = chans
        inner_cin = islifting ? 2 * cin : 4 * cin
        inner_cout = isprojecting ? 2 * cout : 4 * cout

        # Inner conv
        conv = Conv(k, inner_cin => inner_cout, activation; kwargs...)
        new{typeof(conv)}(islifting, isprojecting, cin, cout, conv)
    end
end

## Pretty printing
function Base.show(io::IO, gc::GroupConv2D)
    (; islifting, isprojecting, cin, cout, conv) = gc
    print(io, "GroupConv2D(")
    # print(io, cin, " => ", cout, ", ")
    print(io, conv)
    if islifting || isprojecting
        print(io, "; ")
        islifting && print(io, "islifting = true")
        isprojecting && print(io, "isprojecting = true")
    end
    print(io, ")")
end

function Lux.initialparameters(rng::AbstractRNG, gc::GroupConv2D)
    (; islifting, isprojecting, cin, cout, conv) = gc
    params = Lux.initialparameters(rng, conv)
    (; weight) = params
    dof_cin = islifting || isprojecting ? 2 * cin : 4 * cin
    weight = weight[:, :, 1:dof_cin, 1:cout]
    nx = size(weight, 1)
    if haskey(params, :bias)
        (; bias) = params
        bias = bias[:, :, 1:cout, :]
        (; weight, bias)
    else
        (; weight)
    end
end
Lux.initialstates(rng::AbstractRNG, gc::GroupConv2D) = Lux.initialstates(rng, gc.conv)
# Lux.parameterlength(gc::GroupConv2D) =
#     Lux.parameterlength(gc.conv) % 4
Lux.statelength(gc::GroupConv2D) = Lux.statelength(gc.conv)

function (gc::GroupConv2D)(x, params, state)
    (; islifting, isprojecting, cin, cout, conv) = gc
    (; kernel_size) = conv
    (; weight) = params
    group = (0, 1, 2, 3)

    # Build correctly rotated weight duplicates
    if islifting
        a = weight[:, :, 1:cin, :]
        b = weight[:, :, cin+1:end, :]
        # a = a - rot2(a, 2)
        # b = b - rot2(b, 2)
        w = map(group) do n
            wx, wy = rot2((a, b), n)
            cat(wx, wy; dims = 3)
        end
        weight = cat(w...; dims = 4)
    elseif isprojecting
        a = weight[:, :, 1:cin, :]
        b = weight[:, :, cin+1:end, :]
        # a = a - rot2(a, 2)
        # b = b - rot2(b, 2)
        w = map(group) do m
            wx, wy = rot2((a, b), m)
            cat(wx, wy; dims = 4)
        end
        weight = cat(w...; dims = 3)
    else
        w = map(group) do n
            w = map(group) do m
                i = mod(n - m, 4)
                rot2(weight[:, :, i*cin+1:(i+1)*cin, :], n)
            end
            cat(w...; dims = 3)
        end
        weight = cat(w...; dims = 4)
    end

    # Bias
    params = if haskey(params, :bias)
        (; bias) = params
        if isprojecting
            bias = cat(bias, bias; dims = 3)
        else
            bias = cat(bias, bias, bias, bias; dims = 3)
        end
        (; weight, bias)
    else
        (; weight)
    end

    conv(x, params, state)
end
