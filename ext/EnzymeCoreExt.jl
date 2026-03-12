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
INS.enzyme_wrap(
    f::Union{typeof(INS.apply_bc_u!),typeof(INS.apply_bc_p!),typeof(INS.apply_bc_temp!)},
) = function wrapped_f(y, x, args...)
    # the boundary condition modifies x which is usually the field that we want to differentiate, so we need to introduce a copy of it and modify it instead
    y .= x
    f(y, args...)
    nothing
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
    AugmentedReturn(primal, nothing, nothing)
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
    nothing, nothing, nothing, nothing
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
    nothing, nothing, nothing, nothing
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
    nothing, nothing, nothing, nothing
end
# COV_EXCL_STOP

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
    setup, psolver, viscosity = params
    p = scalarfield(setup)
    u_bc = copy(u.val)
    INS.apply_bc_u!(u_bc, t.val, setup)
    INS.navierstokes!((; u = dudt.val), (; u = u_bc), t; setup, cache = nothing, viscosity)
    INS.apply_bc_u!(dudt.val, t.val, setup)
    INS.project!(dudt.val, setup; psolver, p)
    AugmentedReturn(nothing, nothing, u_bc)
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
    setup, psolver, viscosity = params
    temp_scalar = scalarfield(setup)
    dp = scalarfield(setup)
    temp_vector = vectorfield(setup)

    # traverse the graph backwards
    # [!] notice that the chain starts from the final value of dudt because it gets modified in place in the forward pass
    dudt.dval .*= dudt.val
    # [!] the minus sign is missing somewhere in the adjoint
    dp .= .-INS.pressuregradient_adjoint!(dp, dudt.dval, setup)

    INS.apply_bc_p_pullback!(dp, t.val, setup)

    INS.poisson!(psolver, dp)
    INS.scalewithvolume!(dp, setup)

    dudt.dval .+= INS.divergence_adjoint!(temp_vector, dp, setup)

    INS.apply_bc_u_pullback!(dudt.dval, t.val, setup)

    fill!(temp_vector, 0)
    u.dval .= INS.convection_adjoint!(temp_vector, dudt.dval, u_bc, setup)
    fill!(temp_vector, 0)
    u.dval .+= INS.diffusion_adjoint!(temp_vector, dudt.dval, setup, viscosity)

    INS.apply_bc_u_pullback!(u.dval, t.val, setup)

    nothing, nothing, nothing, nothing
end
# COV_EXCL_STOP

# COV_EXCL_START
# Wrap a function to return `nothing`, because Enzyme can not handle vector return values.
INS.enzyme_wrap(
    f::Union{
        typeof(INS.divergence!),
        typeof(INS.pressuregradient!),
        typeof(INS.convection!),
        typeof(INS.diffusion!),
        typeof(INS.applygravity!),
        typeof(INS.dissipation!),
        typeof(INS.convection_diffusion_temp!),
        typeof(INS.navierstokes!),
    },
) = function wrapped_f(args...)
    f(args...)
    nothing
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{
        Const{typeof(INS.enzyme_wrap(INS.divergence!))},
        Const{typeof(INS.enzyme_wrap(INS.pressuregradient!))},
        Const{typeof(INS.enzyme_wrap(INS.convection!))},
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
    AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.diffusion!))},
    ::Type{<:Const},
    y::Duplicated,
    u::Duplicated,
    setup::Const,
    viscosity::Const,
)
    primal = func.val(y.val, u.val, setup.val, viscosity.val)
    if overwritten(config)[3]
        tape = copy(u.val)
    else
        tape = nothing
    end
    AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{Const{typeof(INS.enzyme_wrap(INS.applygravity!))}},
    ::Type{<:Const},
    y::Duplicated,
    u::Duplicated,
    setup::Const,
    gdir::Const,
    gravity::Const,
)
    primal = func.val(y.val, u.val, setup.val, gdir.val, gravity.val)
    if overwritten(config)[3]
        tape = copy(u.val)
    else
        tape = nothing
    end
    AugmentedReturn(primal, nothing, tape)
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
    nothing, nothing, nothing
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
    nothing, nothing, nothing
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
    nothing, nothing, nothing
end

function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.diffusion!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    setup::Const,
    viscosity::Const,
)
    adj = zero(u.val)
    INS.diffusion_adjoint!(adj, y.val, setup.val, viscosity.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    nothing, nothing, nothing, nothing
end

function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(INS.enzyme_wrap(INS.applygravity!))},
    dret,
    tape,
    y::Duplicated,
    temp::Duplicated,
    setup::Const,
    gdir::Const,
    gravity::Const,
)
    adj = INS.applygravity_adjoint!(zero(temp.val), y.val, setup.val, gdir.val, gravity.val)
    temp.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    nothing, nothing, nothing, nothing, nothing
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{Const{typeof(INS.enzyme_wrap(INS.dissipation!))}},
    ::Type{<:Const},
    y::Duplicated,
    x1::Duplicated,
    x2::Duplicated,
    setup::Const,
    coeff::Const,
)
    primal = func.val(y.val, x1.val, x2.val, setup.val, coeff.val)
    if overwritten(config)[3]
        tape = copy(x2.val)
    else
        tape = nothing
    end
    AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{Const{typeof(INS.enzyme_wrap(INS.convection_diffusion_temp!))}},
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
    AugmentedReturn(primal, nothing, tape)
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
    coeff::Const,
)
    adj = dissipation_pullback(y.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    nothing, nothing, nothing, nothing
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
INS.enzyme_wrap(f::typeof(INS.poisson!)) = function wrapped_f(p, psolve, d)
    p .= d
    f(psolve, p)
    nothing
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
    AugmentedReturn(primal, nothing, nothing)
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
    nothing, nothing, nothing
end

end
# COV_EXCL_STOP
