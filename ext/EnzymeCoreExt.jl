# COV_EXCL_START
"Add Enzyme rules for IncompressibleNavierStokes."
module EnzymeCoreExt

using IncompressibleNavierStokes
using IncompressibleNavierStokes.KernelAbstractions
using IncompressibleNavierStokes.KernelAbstractions.Extras.LoopInfo: @unroll
using EnzymeCore
using EnzymeCore.EnzymeRules

INS = IncompressibleNavierStokes

# Wrap a function to return `nothing`, because Enzyme can not handle vector return values.
function INS.enzyme_wrap(
    f::Union{typeof(INS.apply_bc_u!),typeof(INS.apply_bc_p!),typeof(INS.apply_bc_temp!)},
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
        Const{typeof(INS.enzyme_wrap(INS.apply_bc_u!))},
        Const{typeof(INS.enzyme_wrap(INS.apply_bc_p!))},
        Const{typeof(INS.enzyme_wrap(INS.apply_bc_temp!))},
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
    func::Const{typeof(INS.enzyme_wrap(INS.apply_bc_u!))},
    dret,
    tape,
    y::Duplicated,
    x::Duplicated,
    t::Const,
    setup::Const,
)
    adj = INS.apply_bc_u_pullback!(x.val, t.val, setup.val)
    x.dval .+= adj
    y.dval .= x.dval # y is a copy of x
    return (nothing, nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.apply_bc_p!))},
    dret,
    tape,
    y::Duplicated,
    x::Duplicated,
    t::Const,
    setup::Const,
)
    adj = INS.apply_bc_p_pullback!(x.val, t.val, setup.val)
    x.dval .+= adj
    y.dval .= x.dval # y is a copy of x
    return (nothing, nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.apply_bc_temp!))},
    dret,
    tape,
    y::Duplicated,
    x::Duplicated,
    t::Const,
    setup::Const,
)
    adj = INS.apply_bc_temp_pullback!(x.val, t.val, setup.val)
    x.dval .+= adj
    y.dval .= x.dval # y is a copy of x
    return (nothing, nothing, nothing, nothing)
end
# COV_EXCL_STOP
#
# COV_EXCL_START
function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.right_hand_side!)},
    ::Type{<:Const},
    dudt::Duplicated,
    u::Duplicated,
    params_ref::Any,
    t::Const,
)
    # this runs function to modify dudt and store the intermediates
    params = params_ref.val[]
    setup = params[1]
    psolver = params[2]
    p = scalarfield(setup)
    u_bc = copy(u.val)
    INS.apply_bc_u!(u_bc, t.val, setup)
    INS.navierstokes!((; u = dudt.val), (; u = u_bc), t, nothing, setup, nothing)
    INS.apply_bc_u!(dudt.val, t.val, setup)
    INS.project!(dudt.val, setup; psolver, p)
    return AugmentedReturn(nothing, nothing, u_bc)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.right_hand_side!)},
    dret,
    u_bc,
    dudt::Duplicated,
    u::Duplicated,
    params_ref::Const,
    t::Const,
)
    # unpack the parameters
    params = params_ref.val[]
    setup = params[1]
    psolver = params[2]
    temp_scalar = scalarfield(setup)
    dp = scalarfield(setup)
    temp_vector = vectorfield(setup)

    # traverse the graph backwards
    # [!] notice that the chain starts from the final value of dudt because it gets modified in place in the forward pass
    dudt.dval .*= dudt.val
    # [!] the minus sign is missing somewhere in the adjoint
    dp .= -INS.applypressure_adjoint!(temp_scalar, dudt.dval, nothing, setup)

    INS.apply_bc_p_pullback!(dp, t.val, setup)

    INS.poisson!(psolver, dp)
    INS.scalewithvolume!(dp, setup)

    dudt.dval .+= INS.divergence_adjoint!(temp_vector, dp, setup)

    INS.apply_bc_u_pullback!(dudt.dval, t.val, setup)

    fill!(temp_vector, 0)
    u.dval .= INS.convection_adjoint!(temp_vector, dudt.dval, u_bc, setup)
    fill!(temp_vector, 0)
    u.dval .+= INS.diffusion_adjoint!(temp_vector, dudt.dval, setup)

    INS.apply_bc_u_pullback!(u.dval, t.val, setup)

    return (nothing, nothing, nothing, nothing)
end
# COV_EXCL_STOP

# COV_EXCL_START
# Wrap a function to return `nothing`, because Enzyme can not handle vector return values.
function INS.enzyme_wrap(
    f::Union{
        typeof(INS.divergence!),
        typeof(INS.pressuregradient!),
        typeof(INS.convection!),
        typeof(INS.diffusion!),
        typeof(INS.gravity!),
        typeof(INS.dissipation!),
        typeof(INS.convection_diffusion_temp!),
        typeof(INS.navierstokes!),
    },
)
    function wrapped_f(args...)
        f(args...)
        return nothing
    end
    return wrapped_f
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{
        Const{typeof(INS.enzyme_wrap(INS.divergence!))},
        Const{typeof(INS.enzyme_wrap(INS.pressuregradient!))},
        Const{typeof(INS.enzyme_wrap(INS.convection!))},
        Const{typeof(INS.enzyme_wrap(INS.diffusion!))},
        Const{typeof(INS.enzyme_wrap(INS.gravity!))},
    },
    ::Type{<:Const},
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    primal = func.val(y.val, u.val, setup.val)
    if overwritten(config)[3]
        tape = copy(u.val)
    else
        tape = nothing
    end
    return AugmentedReturn(primal, nothing, tape)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.divergence!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    adj = vectorfield(setup.val)
    INS.divergence_adjoint!(adj, y.val, setup.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.pressuregradient!))},
    dret,
    tape,
    y::Duplicated,
    p::Duplicated,
    setup::Const,
)
    adj = scalarfield(setup.val)
    INS.pressuregradient_adjoint!(adj, y.val, setup.val)
    p.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.convection!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    adj = zero(u.val)
    INS.convection_adjoint!(adj, y.val, u.val, setup.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.diffusion!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    adj = zero(u.val)
    INS.diffusion_adjoint!(adj, y.val, setup.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end

function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.gravity!))},
    dret,
    tape,
    y::Duplicated,
    temp::Duplicated,
    setup::Const,
)
    (; grid, backend, workgroupsize, temperature) = setup.val
    (; dimension, Δ, N, Iu) = grid
    (; gdir, α2) = temperature
    backend = get_backend(temp.val)
    D = dimension()
    e = INS.Offset(D)
    function gravity_pullback(φ)
        @kernel function g!(tempbar, φbar, valα)
            α = INS.getval(valα)
            J = @index(Global, Cartesian)
            t = zero(eltype(tempbar))
            # 1
            I = J
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]+1] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            # 2
            I = J - e(α)
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            tempbar[J] = t
        end
        tempbar = zero(temp.val)
        g!(backend, workgroupsize)(tempbar, φ, Val(gdir); ndrange = N)
        tempbar
    end
    adj = gravity_pullback(y.val)
    temp.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{
        Const{typeof(INS.enzyme_wrap(INS.dissipation!))},
        Const{typeof(INS.enzyme_wrap(INS.convection_diffusion_temp!))},
    },
    ::Type{<:Const},
    y::Duplicated,
    x1::Duplicated,
    x2::Duplicated,
    setup::Const,
)
    primal = func.val(y.val, x1.val, x2.val, setup.val)
    if overwritten(config)[3]
        tape = copy(x2.val)
    else
        tape = nothing
    end
    return AugmentedReturn(primal, nothing, tape)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.dissipation!))},
    dret,
    tape,
    y::Duplicated,
    d::Duplicated,
    u::Duplicated,
    setup::Const,
)
    (; grid, backend, workgroupsize, visc, temperature) = setup.val
    (; dimension, N, Ip) = grid
    (; α1, γ) = temperature
    D = dimension()
    e = INS.Offset(D)
    @kernel function ∂φ!(ubar, dbar, φbar, d, u, valdims)
        J = @index(Global, Cartesian)
        @unroll for β in INS.getval(valdims)
            # Compute ubar
            a = zero(eltype(u))
            # 1
            I = J + e(β)
            I ∈ Ip && (a += α1 / visc / γ * d[I-e(β), β] / 2)
            # 2
            I = J
            I ∈ Ip && (a += α1 / visc / γ * d[I, β] / 2)
            ubar[J, β] += a

            # Compute dbar
            b = zero(eltype(u))
            # 1
            I = J + e(β)
            I ∈ Ip && (b += α1 / visc / γ * u[I-e(β), β] / 2)
            # 2
            I = J
            I ∈ Ip && (b += α1 / visc / γ * u[I, β] / 2)
            dbar[J, β] += b
        end
    end
    function dissipation_pullback(φbar)
        # Dφ/Du = ∂φ(u, d)/∂u + ∂φ(u, d)/∂d ⋅ ∂d(u)/∂u
        dbar = zero(u.val)
        ubar = zero(u.val)
        ∂φ!(backend, workgroupsize)(ubar, dbar, φbar, d.val, u.val, Val(1:D); ndrange = N)
        INS.diffusion_adjoint!(ubar, dbar, setup.val)
        ubar
    end
    adj = dissipation_pullback(y.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.convection_diffusion_temp!))},
    dret,
    tape,
    y::Duplicated,
    temp::Duplicated,
    u::Duplicated,
    setup::Const,
)
    @error "convection_diffusion_temp Enzyme-AD not yet implemented"
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{Const{typeof(INS.enzyme_wrap(INS.navierstokes!))}},
    ::Type{<:Const},
    y::Duplicated,
    x1::Duplicated,
    x2::Duplicated,
    x3::Duplicated,
    t::Const,
    setup::Const,
)
    @error "navierstokes! Enzyme-AD not yet implemented"
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.navierstokes!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    temp::Duplicated,
    t::Const,
    setup::Const,
)
    @error "navierstokes! Enzyme-AD not yet implemented"
end
# COV_EXCL_STOP

# COV_EXCL_START
# Wrap a function to return `nothing`, because Enzyme can not handle vector return values.
function INS.enzyme_wrap(f::typeof(INS.poisson!))
    function wrapped_f(p, psolve, d)
        p .= d
        f(psolve, p)
        return nothing
    end
    return wrapped_f
end
function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.poisson!))},
    ::Type{<:Const},
    y::Duplicated,
    psolver::Const,
    div::Duplicated,
)
    primal = func.val(y.val, psolver.val, div.val)
    return AugmentedReturn(primal, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.poisson!))},
    dret,
    tape,
    y::Duplicated,
    psolver::Const,
    div::Duplicated,
)
    auto_adj = copy(y.val)
    func.val(auto_adj, psolver.val, y.val)
    div.dval .+= auto_adj .* y.dval
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end

end
# COV_EXCL_STOP
