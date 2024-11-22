"Boundary condition for one side of the domain."
abstract type AbstractBC end

"Periodic boundary conditions. Must be periodic on both sides."
struct PeriodicBC <: AbstractBC end

"""
Dirichlet boundary conditions for the velocity, where `u[1] = (x..., t) ->
u1_BC` up to `u[d] = (x..., t) -> ud_BC`, where `d` is the dimension.

When `u` is `nothing`, then the boundary conditions are
no slip boundary conditions, where all velocity components are zero.
"""
struct DirichletBC{U} <: AbstractBC
    "Boundary condition"
    u::U
end

DirichletBC() = DirichletBC(nothing)

"""
Symmetric boundary conditions.
The parallel velocity and pressure is the same at each side of the boundary.
The normal velocity is zero.
"""
struct SymmetricBC <: AbstractBC end

"""
Pressure boundary conditions.
The pressure is prescribed on the boundary (usually an "outlet").
The velocity has zero Neumann conditions.

Note: Currently, the pressure is prescribed with the constant value of
zero on the entire boundary.
"""
struct PressureBC <: AbstractBC end

# Pad volume boundary coordinate vector with ghost coordinates
function padghost! end

# Add opposite boundary ghost volume
padghost!(::PeriodicBC, x, isright) =
    if isright
        Δx_a = x[2] - x[1]
        push!(x, x[end] + Δx_a)
    else
        Δx_b = x[end] - x[end-1]
        pushfirst!(x, x[1] - Δx_b)
    end

# Add infinitely thin boundary volume
padghost!(::DirichletBC, x, isright) = isright ? push!(x, x[end]) : pushfirst!(x, x[1])

# Duplicate boundary volume
padghost!(::SymmetricBC, x, isright) =
    isright ? push!(x, x[end] + (x[end] - x[end-1])) : pushfirst!(x, x[1] - (x[2] - x[1]))

# Add infinitely thin boundary volume
# On the left, we need to add two ghost volumes to have a normal component at
# the left of the first ghost volume
padghost!(::PressureBC, x, isright) = isright ? push!(x, x[end]) : pushfirst!(x, x[1], x[1])

"""
    $FUNCTIONNAME(bc, isnormal, isright)

Number of non-DOF velocity components at boundary.
If `isnormal`, then the velocity is normal to the boundary, else parallel.
If `isright`, it is at the end/right/rear/top boundary, otherwise beginning.
"""
function offset_u end

"""
    $FUNCTIONNAME(bc, isnormal, isright)

Number of non-DOF pressure components at boundary.
"""
function offset_p end

offset_u(::PeriodicBC, isright, isnormal) = 1
offset_p(::PeriodicBC, isright) = 1

offset_u(::DirichletBC, isright, isnormal) = 1 + isright * isnormal
offset_p(::DirichletBC, isright) = 1

offset_u(::SymmetricBC, isright, isnormal) = 1 + isright * isnormal
offset_p(::SymmetricBC, isright) = 1

offset_u(::PressureBC, isright, isnormal) = 1 + !isright * !isnormal
offset_p(::PressureBC, isright) = 1 + !isright

"""
Get boundary indices of boundary layer normal to `β`.
The `CartesianIndices` given by `I` should contain those of the inner DOFs,
typically `Ip` or `Iu[α]`.
The boundary layer is then just outside those.
"""
function boundary(β, N, I, isright)
    D = length(N)
    Iβ = I.indices[β]
    i = isright ? Iβ[end] + 1 : Iβ[1] - 1
    ranges = ntuple(α -> α == β ? (i:i) : (1:N[α]), D)
    CartesianIndices(ranges)
end

"Apply velocity boundary conditions (differentiable version)."
apply_bc_u(u, t, setup; kwargs...) = apply_bc_u!(copy(u), t, setup; kwargs...)

"Apply pressure boundary conditions (differentiable version)."
apply_bc_p(p, t, setup; kwargs...) = apply_bc_p!(copy(p), t, setup; kwargs...)

"Apply temperature boundary conditions (differentiable version)."
apply_bc_temp(temp, t, setup; kwargs...) = apply_bc_temp!(copy(temp), t, setup; kwargs...)

ChainRulesCore.rrule(::typeof(apply_bc_u), u, t, setup; kwargs...) = (
    apply_bc_u(u, t, setup; kwargs...),
    # With respect to (apply_bc_u, u, t, setup)
    φbar -> (
        NoTangent(),
        # Important: identity operator should be part of `apply_bc_u_pullback`,
        # but is actually implemented via the `copy` below instead.
        apply_bc_u_pullback!(copy(unthunk(φbar)), t, setup; kwargs...),
        NoTangent(),
        NoTangent(),
    ),
)
ChainRulesCore.rrule(::typeof(apply_bc_p), p, t, setup) = (
    apply_bc_p(p, t, setup),
    # With respect to (apply_bc_p, p, t, setup)
    φbar -> (
        NoTangent(),
        apply_bc_p_pullback!(
            # Important: identity operator should be part of `apply_bc_p_pullback`,
            # but is actually implemented via the `copy` below instead.
            copy(unthunk(φbar)),
            t,
            setup,
        ),
        NoTangent(),
        NoTangent(),
    ),
)
ChainRulesCore.rrule(::typeof(apply_bc_temp), temp, t, setup) = (
    apply_bc_temp(temp, t, setup),
    # With respect to (apply_bc_temp, temp, t, setup)
    φbar -> (
        NoTangent(),
        apply_bc_temp_pullback!(
            # Important: identity operator should be part of `apply_bc_temp_pullback`,
            # but is actually implemented via the `copy` below instead.
            copy(unthunk(φbar)),
            t,
            setup,
        ),
        NoTangent(),
        NoTangent(),
    ),
)
"Apply velocity boundary conditions (in-place version)."
function apply_bc_u!(u, t, setup; kwargs...)
    (; boundary_conditions) = setup
    D = setup.grid.dimension()
    for β = 1:D
        apply_bc_u!(boundary_conditions[β][1], u, β, t, setup; isright = false, kwargs...)
        apply_bc_u!(boundary_conditions[β][2], u, β, t, setup; isright = true, kwargs...)
    end
    u
end

function apply_bc_u_pullback!(φbar, t, setup; kwargs...)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_u_pullback!(
            boundary_conditions[β][1],
            φbar,
            β,
            t,
            setup;
            isright = false,
            kwargs...,
        )
        apply_bc_u_pullback!(
            boundary_conditions[β][2],
            φbar,
            β,
            t,
            setup;
            isright = true,
            kwargs...,
        )
    end
    φbar
end

"Apply pressure boundary conditions (in-place version)."
function apply_bc_p!(p, t, setup; kwargs...)
    (; boundary_conditions, grid) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_p!(boundary_conditions[β][1], p, β, t, setup; isright = false)
        apply_bc_p!(boundary_conditions[β][2], p, β, t, setup; isright = true)
    end
    p
end

function apply_bc_p_pullback!(φbar, t, setup; kwargs...)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_p_pullback!(
            boundary_conditions[β][1],
            φbar,
            β,
            t,
            setup;
            isright = false,
            kwargs...,
        )
        apply_bc_p_pullback!(
            boundary_conditions[β][2],
            φbar,
            β,
            t,
            setup;
            isright = true,
            kwargs...,
        )
    end
    φbar
end

"Apply temperature boundary conditions (in-place version)."
function apply_bc_temp!(temp, t, setup; kwargs...)
    (; temperature, grid) = setup
    (; boundary_conditions) = temperature
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_temp!(boundary_conditions[β][1], temp, β, t, setup; isright = false)
        apply_bc_temp!(boundary_conditions[β][2], temp, β, t, setup; isright = true)
    end
    temp
end

function apply_bc_temp_pullback!(φbar, t, setup; kwargs...)
    (; temperature, grid) = setup
    (; boundary_conditions) = temperature
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_temp_pullback!(
            boundary_conditions[β][1],
            φbar,
            β,
            t,
            setup;
            isright = false,
            kwargs...,
        )
        apply_bc_temp_pullback!(
            boundary_conditions[β][2],
            φbar,
            β,
            t,
            setup;
            isright = true,
            kwargs...,
        )
    end
    φbar
end

function apply_bc_u!(::PeriodicBC, u, β, t, setup; isright, kwargs...)
    isright && return u # We do both in one go for "left"
    (; dimension, N, Iu) = setup.grid
    D = dimension()
    eβ = Offset(D)(β)
    Ia = boundary(β, N, Iu[1], false)
    Ib = boundary(β, N, Iu[1], true)
    Ja = Ia .+ eβ
    Jb = Ib .- eβ
    @. u[Ia, :] = u[Jb, :]
    @. u[Ib, :] = u[Ja, :]
    u
end

function apply_bc_u_pullback!(::PeriodicBC, φbar, β, t, setup; isright, kwargs...)
    isright && return φbar # We do both in one go for "left"
    (; dimension, N, Iu) = setup.grid
    D = dimension()
    eβ = Offset(D)(β)
    Ia = boundary(β, N, Iu[1], false)
    Ib = boundary(β, N, Iu[1], true)
    Ja = Ia .+ eβ
    Jb = Ib .- eβ
    @. φbar[Jb, :] += φbar[Ia, :]
    @. φbar[Ja, :] += φbar[Ib, :]
    @. φbar[Ia, :] = 0
    @. φbar[Ib, :] = 0
    φbar
end

function apply_bc_p!(bc::PeriodicBC, p, β, t, setup; isright, kwargs...)
    isright && return p # We do both in one go for "left"
    (; dimension, N, Ip) = setup.grid
    D = dimension()
    eβ = Offset(D)(β)
    Ia = boundary(β, N, Ip, false)
    Ib = boundary(β, N, Ip, true)
    Ja = Ia .+ eβ
    Jb = Ib .- eβ
    @. p[Ia] = p[Jb]
    @. p[Ib] = p[Ja]
    p
end

function apply_bc_p_pullback!(::PeriodicBC, φbar, β, t, setup; isright, kwargs...)
    isright && return φbar # We do both in one go for "left"
    (; dimension, N, Ip) = setup.grid
    D = dimension()
    eβ = Offset(D)(β)
    Ia = boundary(β, N, Ip, false)
    Ib = boundary(β, N, Ip, true)
    Ja = Ia .+ eβ
    Jb = Ib .- eβ
    @. φbar[Jb] += φbar[Ia]
    @. φbar[Ja] += φbar[Ib]
    @. φbar[Ia] = 0
    @. φbar[Ib] = 0
    φbar
end

apply_bc_temp!(bc::PeriodicBC, temp, β, t, setup; isright, kwargs...) =
    apply_bc_p!(bc, temp, β, t, setup; isright, kwargs...)

apply_bc_temp_pullback!(bc::PeriodicBC, φbar, β, t, setup; isright, kwargs...) =
    apply_bc_p_pullback!(bc, φbar, β, t, setup; isright, kwargs...)

function apply_bc_u!(bc::DirichletBC, u, β, t, setup; isright, dudt = false, kwargs...)
    (; dimension, N, xu, Iu) = setup.grid
    D = dimension()
    bcfunc = if isnothing(bc.u)
        Returns(0)
    elseif bc.u isa Tuple
        (α, args...) -> dudt ? zero(bc.u[α]) : bc.u[α]
    elseif dudt
        # Use central difference to approximate dudt
        h = sqrt(eps(eltype(u))) / 2
        function (args...)
            args..., t = args
            (bc.u(args..., t + h) - bc.u(args..., t - h)) / 2h
        end
    else
        bc.u
    end
    for α = 1:D
        I = boundary(β, N, Iu[α], isright)
        xI = ntuple(
            γ -> reshape(
                xu[α][γ][I.indices[γ]],
                ntuple(Returns(1), γ - 1)...,
                :,
                ntuple(Returns(1), D - γ)...,
            ),
            D,
        )
        u[I, α] .= bcfunc.(α, xI..., t)
    end
    u
end

function apply_bc_u_pullback!(::DirichletBC, φbar, β, t, setup; isright, kwargs...)
    (; dimension, N, Iu) = setup.grid
    D = dimension()
    for α = 1:D
        I = boundary(β, N, Iu[α], isright)
        φbar[I, α] .= 0
    end
    φbar
end

function apply_bc_p!(::DirichletBC, p, β, t, setup; isright, kwargs...)
    (; dimension, N, Ip) = setup.grid
    D = dimension()
    e = Offset(D)
    I = boundary(β, N, Ip, isright)
    J = isright ? I .- e(β) : I .+ e(β)
    @. p[I] = p[J]
    p
end

function apply_bc_p_pullback!(::DirichletBC, φbar, β, t, setup; isright, kwargs...)
    (; dimension, N, Ip) = setup.grid
    D = dimension()
    e = Offset(D)
    I = boundary(β, N, Ip, isright)
    J = isright ? I .- e(β) : I .+ e(β)
    @. φbar[J] += φbar[I]
    φbar[I] .= 0
    φbar
end

function apply_bc_temp!(bc::DirichletBC, temp, β, t, setup; isright, kwargs...)
    (; dimension, N, Ip, xp) = setup.grid
    D = dimension()
    I = boundary(β, N, Ip, isright)
    bcfunc = if isnothing(bc.u)
        Returns(0)
    elseif bc.u isa Number
        Returns(bc.u)
    else
        bc.u
    end
    xI = ntuple(α -> reshape(xp[α][I.indices[α]], ntuple(Returns(1), α - 1)..., :), D)
    temp[I] .= bcfunc.(xI..., t)
    temp
end

function apply_bc_temp_pullback!(::DirichletBC, φbar, β, t, setup; isright, kwargs...)
    (; N, Ip) = setup.grid
    I = boundary(β, N, Ip, isright)
    φbar[I] .= 0
    φbar
end

function apply_bc_u!(::SymmetricBC, u, β, t, setup; isright, kwargs...)
    (; dimension, Nu, Iu) = setup.grid
    D = dimension()
    e = Offset(D)
    for α = 1:D
        if α != β
            I = boundary(β, Nu[α], Iu[α], isright)
            J = isright ? I .- e(β) : I .+ e(β)
            @. u[I, α] = u[J, α]
        end
    end
    u
end

function apply_bc_u_pullback!(::SymmetricBC, φbar, β, t, setup; isright, kwargs...)
    (; dimension, Nu, Iu) = setup.grid
    D = dimension()
    e = Offset(D)
    for α = 1:D
        if α != β
            I = boundary(β, Nu[α], Iu[α], isright)
            J = isright ? I .- e(β) : I .+ e(β)
            @. φbar[J, α] += φbar[I, α]
            @. φbar[I, α] = 0
        end
    end
    φbar
end

function apply_bc_p!(::SymmetricBC, p, β, t, setup; isright, kwargs...)
    (; dimension, N, Ip) = setup.grid
    D = dimension()
    e = Offset(D)
    I = boundary(β, N, Ip, isright)
    J = isright ? I .- e(β) : I .+ e(β)
    @. p[I] = p[J]
    p
end

function apply_bc_p_pullback!(::SymmetricBC, φbar, β, t, setup; isright, kwargs...)
    (; dimension, N, Ip) = setup.grid
    D = dimension()
    e = Offset(D)
    I = boundary(β, N, Ip, isright)
    J = isright ? I .- e(β) : I .+ e(β)
    @. φbar[J] += φbar[I]
    @. φbar[I] = 0
    φbar
end

apply_bc_temp!(bc::SymmetricBC, temp, β, t, setup; isright, kwargs...) =
    apply_bc_p!(bc, temp, β, t, setup; isright, kwargs...)

apply_bc_temp_pullback!(bc::SymmetricBC, φbar, β, t, setup; isright, kwargs...) =
    apply_bc_p_pullback!(bc, φbar, β, t, setup; isright, kwargs...)

function apply_bc_u!(bc::PressureBC, u, β, t, setup; isright, kwargs...)
    (; dimension, N, Iu) = setup.grid
    D = dimension()
    e = Offset(D)
    for α = 1:D
        I = boundary(β, N, Iu[α], isright)
        J = isright ? I .- e(β) : I .+ e(β)
        @. u[I, α] = u[J, α]
    end
    u
end

function apply_bc_u_pullback!(::PressureBC, φbar, β, t, setup; isright, kwargs...)
    (; dimension, N, Iu) = setup.grid
    D = dimension()
    e = Offset(D)
    for α = 1:D
        I = boundary(β, N, Iu[α], isright)
        J = isright ? I .- e(β) : I .+ e(β)
        @. φbar[J, α] += φbar[I, α]
        @. φbar[I, α] = 0
    end
    φbar
end

function apply_bc_p!(bc::PressureBC, p, β, t, setup; isright, kwargs...)
    (; N, Ip) = setup.grid
    I = boundary(β, N, Ip, isright)
    p[I] .= 0
    p
end

function apply_bc_p_pullback!(::PressureBC, φbar, β, t, setup; isright, kwargs...)
    (; N, Ip) = setup.grid
    I = boundary(β, N, Ip, isright)
    φbar[I] .= 0
    φbar
end

# Symmetric BC for temperature
apply_bc_temp!(bc::PressureBC, temp, β, t, setup; isright, kwargs...) =
    apply_bc_temp!(SymmetricBC(), temp, β, t, setup; isright, kwargs...)

apply_bc_temp_pullback!(bc::PressureBC, φbar, β, t, setup; isright, kwargs...) =
    apply_bc_p_pullback!(SymmetricBC(), φbar, β, t, setup; isright, kwargs...)

# COV_EXCL_START
# Wrap a function to return `nothing`, because Enzyme can not handle vector return values.
function enzyme_wrap(
    f::Union{typeof(apply_bc_u!),typeof(apply_bc_p!),typeof(apply_bc_temp!)},
)
    # the boundary condition modifies x which is usually the field that we want to differentiate, so we need to introduce a copy of it and modify it instead
    function wrapped_f(y, x, args...)
        y .= x
        f(y, args...)
        return nothing
    end
    return wrapped_f
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{
        Const{typeof(enzyme_wrap(apply_bc_u!))},
        Const{typeof(enzyme_wrap(apply_bc_p!))},
        Const{typeof(enzyme_wrap(apply_bc_temp!))},
    },
    ::Type{<:Const},
    y::Duplicated,
    x::Duplicated,
    t::Const,
    setup::Const,
)
    primal = func.val(y.val, x.val, t.val, setup.val)
    return AugmentedReturn(primal, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(apply_bc_u!))},
    dret,
    tape,
    y::Duplicated,
    x::Duplicated,
    t::Const,
    setup::Const,
)
    adj = apply_bc_u_pullback!(x.val, t.val, setup.val)
    x.dval .+= adj
    y.dval .= x.dval # y is a copy of x
    return (nothing, nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(apply_bc_p!))},
    dret,
    tape,
    y::Duplicated,
    x::Duplicated,
    t::Const,
    setup::Const,
)
    adj = apply_bc_p_pullback!(x.val, t.val, setup.val)
    x.dval .+= adj
    y.dval .= x.dval # y is a copy of x
    return (nothing, nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(apply_bc_temp!))},
    dret,
    tape,
    y::Duplicated,
    x::Duplicated,
    t::Const,
    setup::Const,
)
    adj = apply_bc_temp_pullback!(x.val, t.val, setup.val)
    x.dval .+= adj
    y.dval .= x.dval # y is a copy of x
    return (nothing, nothing, nothing, nothing)
end
# COV_EXCL_STOP
