
function create_right_hand_side(setup, psolver) 
    function right_hand_side(u, param, t)
        F = zeros(size(u))
        u = apply_bc_u(u, t, setup)
        #F = convection(u, setup) .+ diffusion(u, setup)
        F = momentum(u, nothing, t, setup)
        F = apply_bc_u(F, t, setup; dudt = true)
        FP = project(F, setup; psolver)
        #p = divergence(F, setup)
        #p = scalewithvolume(p, setup)
        #p = poisson(psolver, p)
        #p = apply_bc_p(p, t, setup)
        #G = pressuregradient(p, setup)
        #F .- G
    end
end


function right_hand_side!(dudt, u, params_ref, t)
    params = params_ref[]
    setup = params[1]
    psolver = params[2]
    p = scalarfield(setup)
    # [!]*** be careful to not touch u in this function!
    temp_vector = copy(u)
    apply_bc_u!(temp_vector, t, setup)
    momentum!(dudt, temp_vector, nothing, t, setup)
    #fill!(dudt, 0)
    #convectiondiffusion!(dudt, temp_vector, setup)
    apply_bc_u!(dudt, t, setup)
    project!(dudt, setup; psolver, p)
    #divergence!(p, dudt, setup)
    #scalewithvolume!(p, setup)
    #poisson!(psolver, p)
    #apply_bc_p!(p, t, setup)
    #applypressure!(dudt, p, setup)
    return nothing
end


using Enzyme
import .EnzymeRules: reverse, augmented_primal
using .EnzymeRules
function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(right_hand_side!)}, ::Type{<:Const}, dudt::Duplicated, u::Duplicated, params_ref::Any, t::Const)
    # this runs function to modify dudt and store the intermediates
    params = params_ref.val[]
    setup = params[1]
    psolver = params[2]
    p = scalarfield(setup)
    u_bc = copy(u.val)
    apply_bc_u!(u_bc, t.val, setup)
    momentum!(dudt.val, u_bc, nothing, t, setup)
    #fill!(dudt.val, 0)
    #convectiondiffusion!(dudt.val, u_bc, setup)
    apply_bc_u!(dudt.val, t.val, setup)
    project!(dudt.val, setup; psolver, p)
    #divergence!(p, dudt.val, setup)
    #scalewithvolume!(p, setup)
    #poisson!(psolver, p)
    #apply_bc_p!(p, t.val, setup)
    #applypressure!(dudt.val, p, setup)
    return AugmentedReturn(nothing, nothing, u_bc)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(right_hand_side!)}, dret, u_bc, dudt::Duplicated, u::Duplicated, params_ref::Const, t::Const)
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
    dp .= - applypressure_adjoint!(temp_scalar, dudt.dval, nothing, setup)

    apply_bc_p_pullback!(dp, t.val, setup)

    poisson!(psolver,dp)
    scalewithvolume!(dp, setup)

    dudt.dval .+= divergence_adjoint!(temp_vector, dp, setup)

    apply_bc_u_pullback!(dudt.dval, t.val, setup)

    fill!(temp_vector, 0)
    u.dval .= convection_adjoint!(temp_vector, dudt.dval, u_bc, setup)
    fill!(temp_vector, 0)
    u.dval .+= diffusion_adjoint!(temp_vector, dudt.dval, setup)

    apply_bc_u_pullback!(u.dval, t.val, setup)
    
    return (nothing, nothing, nothing, nothing)
end