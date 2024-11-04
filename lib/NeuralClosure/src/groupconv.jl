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
        j = (1:ny)'
    elseif r == 1
        i = (1:nx)'
        j = ny:-1:1
    elseif r == 2
        i = nx:-1:1
        j = (ny:-1:1)'
    elseif r == 3
        i = (nx:-1:1)'
        j = 1:ny
    end
    I = CartesianIndex.(i, j)
    chans = fill(:, length(s))
    u[I, chans...]
end

"Rotate vector fields `[ux;;; uy]`"
function rot2(u::Tuple{T,T}, r) where {T}
    # ux, uy = eachslice(u; dims = ndims(u))
    ux, uy = u
    r = mod(r, 4)
    rx = rot2(ux, r)
    ry = rot2(uy, r)
    ru = if r == 0
        (rx, ry)
    elseif r == 1
        (-ry, rx)
    elseif r == 2
        (-rx, -ry)
    elseif r == 3
        (ry, -rx)
    end
    ru
end

"Rotate vector fields `[ux;;; uy]`"
function vecrot2(u, r)
    # ux, uy = eachslice(u; dims = ndims(u))
    ux, uy = u[:, :, 1], u[:, :, 2]
    r = mod(r, 4)
    rx = rot2(ux, r)
    ry = rot2(uy, r)
    ru = if r == 0
        (rx, ry)
    elseif r == 1
        (-ry, rx)
    elseif r == 2
        (-rx, -ry)
    elseif r == 3
        (ry, -rx)
    end
    stack(ru)
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

"Rotate staggered grid velocity field. See also [`rot2`](@ref)."
function rot2stag(u, g)
    g = mod(g, 4)
    u = vecrot2(u, g)
    # ux, uy = eachslice(u; dims = ndims(u))
    ux, uy = u[:, :, 1], u[:, :, 2]
    if g in (1, 2)
        ux = circshift(ux, -1)
        ux[end, :] .= ux[2, :]
    end
    if g in (2, 3)
        uy = circshift(uy, (0, -1))
        uy[:, end] .= uy[:, 2]
    end
    cat(ux, uy; dims = 3)
end

"""
Group-equivariant convolutional layer -- with respect to the p4 group.
The layer is equivariant to rotations and translations of the input
vector field.

The `kwargs` are passed to the `Conv` layer.

The layer has three variants:

- If `islifting` then it lifts a vector input `(u1, u2)` into a rotation-state vector `(v1, v2, v3, v4)`.
- If `isprojecting`, it projects a rotation-state vector `(u1, u2, u3, v4)` into a vector `(v1, v2)`.
- Otherwise, it cyclically transforms the rotation-state vector `(u1, u2, u3, u4)` into a new rotation-state vector `(v1, v2, v3, v4)`.
"""
struct GroupConv2D{C} <: Lux.AbstractLuxLayer
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
    if islifting || isprojecting
        weight = (;
            w1 = weight[:, :, 0*cin+1:1*cin, 1:cout],
            w2 = weight[:, :, 1*cin+1:2*cin, 1:cout],
        )
    else
        weight = (;
            w1 = weight[:, :, 0*cin+1:1*cin, 1:cout],
            w2 = weight[:, :, 1*cin+1:2*cin, 1:cout],
            w3 = weight[:, :, 2*cin+1:3*cin, 1:cout],
            w4 = weight[:, :, 3*cin+1:4*cin, 1:cout],
        )
    end
    if Lux.has_bias(conv)
        (; bias) = params
        bias = bias[1:cout]
        (; weight, bias)
    else
        (; weight)
    end
end

Lux.initialstates(rng::AbstractRNG, gc::GroupConv2D) = Lux.initialstates(rng, gc.conv)

function Lux.parameterlength(gc::GroupConv2D)
    (; islifting, isprojecting, cin, cout, conv) = gc
    (; kernel_size) = conv
    nn = islifting || isprojecting ? 2 : 4
    n = nn * prod(kernel_size) * cin * cout
    n += Lux.has_bias(conv) * cout
end

Lux.statelength(gc::GroupConv2D) = Lux.statelength(gc.conv)

function (gc::GroupConv2D)(x, params, state)
    (; islifting, isprojecting, cin, cout, conv) = gc
    (; kernel_size) = conv
    (; weight) = params
    group = (0, 1, 2, 3)

    # Build correctly rotated weight duplicates
    weight = if islifting
        (; w1, w2) = weight
        # a = a - rot2(a, 2)
        # b = b - rot2(b, 2)
        cat(map(group) do n
            wx, wy = rot2((w1, w2), n)
            cat(wx, wy; dims = 3)
        end...; dims = 4)
    elseif isprojecting
        (; w1, w2) = weight
        # a = a - rot2(a, 2)
        # b = b - rot2(b, 2)
        cat(map(group) do m
            wx, wy = rot2((w1, w2), m)
            cat(wx, wy; dims = 4)
        end...; dims = 3)
    else
        (; w1, w2, w3, w4) = weight
        w = (w1, w2, w3, w4)
        cat(map(group) do n
            cat(map(group) do m
                i = mod(n - m, 4) + 1
                rot2(w[i], n)
            end...; dims = 3)
        end...; dims = 4)
    end

    # Bias
    params = if Lux.has_bias(conv)
        (; bias) = params
        bias = if isprojecting
            vcat(bias, bias)
        else
            vcat(bias, bias, bias, bias)
        end
        (; weight, bias)
    else
        (; weight)
    end

    conv(x, params, state)
end

"""
Create CNN closure model. Return a tuple `(closure, θ)` where `θ` are the initial
parameters and `closure(u, θ)` predicts the commutator error.
"""
function gcnn(; setup, radii, channels, activations, use_bias, rng = Random.default_rng())
    r, c, σ, b = radii, channels, activations, use_bias
    (; grid) = setup
    (; dimension, x) = grid
    D = dimension()
    T = eltype(x[1])

    # Weight initializer
    glorot_uniform_T(rng::AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    # Add input channel size
    c = [1; c]

    # Add padding so that output has same shape as commutator error
    padder = ntuple(α -> (u -> pad_circular(u, sum(r); dims = α)), D)

    # Some convolutional layers
    gconvs = map(
        i -> GroupConv2D(
            ntuple(α -> 2r[i] + 1, D),
            c[i] => c[i+1],
            σ[i];
            use_bias = b[i],
            init_weight = glorot_uniform_T,
            islifting = i == 1,
            isprojecting = i == length(r),
        ),
        eachindex(r),
    )

    # Create group-convolutional closure model
    create_closure(collocate, padder, gconvs..., decollocate; rng)
end
