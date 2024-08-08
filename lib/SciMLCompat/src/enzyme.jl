module INSenzyme
import IncompressibleNavierStokes 

""" 
This module wraps some of the functions of the IncompressibleNavierStokes module
in order to make them compatible with Enzyme AD.

There are also some tests to check if the redefined functions are faster than the original ones and if they are differentiable.
"""

using SparseArrays
using LinearAlgebra

"""
    __get_enz_bc_u!(_backend, setup)

This function is used to precompile the function that applies the boundary conditions on u.
It wraps everything such that the only thing that has to be passed from outside is the u array.
This makes it easy for Enzyme to differentiate the function.
"""
function _get_enz_bc_u!(_backend, setup)
    (; boundary_conditions, grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    for β = 1:D
        @assert boundary_conditions[β][1] isa PeriodicBC "Only PeriodicBC implemented"
    end

    @kernel function _bc_a!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I] = u[α][I+(N[β]-2)*e(β)]
    end
    @kernel function _bc_b!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I+(N[β]-1)*e(β)] = u[α][I+e(β)]
    end

    function bc_u!(u, β; isright, )
        ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
        if isright
            for α = 1:D
                _bc_b!(_backend, workgroupsize)(u, Val(α), Val(β); ndrange)
            end
        else
            for α = 1:D
                _bc_a!(_backend, workgroupsize)(u, Val(α), Val(β); ndrange)
            end
        end
        
    end

    function bc(u)
        for β = 1:D
            bc_u!(u, β; isright = false)
            bc_u!(u, β; isright = true)
        end
    end
end

"""
    _get_enz_bc_p!(_backend, setup)

This function is used to precompile the function that applies the boundary conditions on p.
"""
function _get_enz_bc_p!(_backend, setup) 
    (; boundary_conditions, grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    for β = 1:D
        @assert boundary_conditions[β][1] isa PeriodicBC "Only PeriodicBC implemented"
    end
    
    function bc_p!(p, β; isright)
        @kernel function _bc_a(p, ::Val{β}) where {β}
            I = @index(Global, Cartesian)
            p[I] = p[I+(N[β]-2)*e(β)]
        end
        @kernel function _bc_b(p, ::Val{β}) where {β}
            I = @index(Global, Cartesian)
            p[I+(N[β]-1)*e(β)] = p[I+e(β)]
        end
        ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
        if isright
            _bc_b(_backend, workgroupsize)(p, Val(β); ndrange)
        else
            _bc_a(_backend, workgroupsize)(p, Val(β); ndrange)
        end
    end

    function bc!(p)
        for β = 1:D
            bc_p!(p, β; isright = false)
            bc_p!(p, β; isright = true)
        end
    end
end

"""
    _cache_psolver(grid, setup)

This function is used to precompile the parameters of the psolver function.
Its implementation is similar to the one in IncompressibleNavierStokes.
"""
function _cache_psolver(::Array, setup)
    (; grid, boundary_conditions) = setup
    (; x, Np, Ip) = grid
    T = eltype(x[1])
    L = laplacian_mat(setup)
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
    if isdefinite
        println("Definite")
        # No extra DOF
        T = Float64 # This is currently required for SuiteSparse LU
        ftemp = zeros(T, prod(Np))
        ptemp = zeros(T, prod(Np))
        viewrange = (:)
        fact = factorize(L)
    else
        println("Indefinite")
        # With extra DOF
        ftemp = zeros(T, prod(Np) + 1)
        ptemp = zeros(T, prod(Np) + 1)
        e = ones(T, size(L, 2))
        L = [L e; e' 0]
        maximum(L - L') < sqrt(eps(T)) || error("Matrix not symmetric")
        L = @. (L + L') / 2
        viewrange = 1:prod(Np)
        fact = ldlt(L)
    end
    return fact, viewrange, Ip
end

"""
    _get_enz_psolver(setup)

This function is used to precompile the psolver function.
In particular, it has to wrap fact, viewrange, and Ip such that Enzyme does not have to allocate memory for them.
"""

function _get_enz_psolver(setup)
    fact, viewrange, Ip = _cache_psolver(setup.grid.x[1], setup)
    function psolver(p, f, ftemp, ptemp)
        copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
        ptemp .= fact \ ftemp
        copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
        nothing
    end
end

"""
    get_enz_div!(_backend, setup)

This function is used to precompile the function that computes the divergence in place.
In particular, it precompiles Δ inside the function such that Enzyme does not try to Dual() it and fail because it is a tuple.
Also, the automatic differentiations requires a cache d to be preallocated and passed as an argument, on top of div.
"""

function _get_enz_div!(_backend, setup) 
    (; grid, workgroupsize) = setup
    (; Δ, Ip, Np) = grid
    D = length(u)
    e = Offset{D}()
    @kernel function div!(div, u, I0, d, Δ)
        I = @index(Global, Cartesian)
        I = I + I0
        for α = 1:D
            d[I] += (u[α][I] - u[α][I-e(α)]) / Δ[α][I[α]]
        end
        div[I] = d[I]
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    function d!(div, u, d)
        # set the temporary array to zero
        @. d *= 0
        # It requires Δ to be passed from outside to comply with Enzyme
        div!(_backend, workgroupsize)(div, u, I0, d, Δ; ndrange = Np)
        nothing
    end
end


"""
    _get_enz_applypressure!(_backend, setup)

This function is used to precompile the function that applies the pressure in place.
Here Δu is precompiled into the function thus being hidden from Enzyme.
"""

function _get_enz_applypressure!(_backend, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δu, Nu, Iu) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function apply!(u, p, ::Val{α}, I0, Δu) where {α}
        I = @index(Global, Cartesian)
        I = I0 + I
        u[α][I] -= (p[I+e(α)] - p[I]) / Δu[α][I[α]]
    end
    function ap!(u, p)
        for α = 1:D
            I0 = first(Iu[α])
            I0 -= oneunit(I0)
            apply!(_backend, workgroupsize)(u, p, Val(α), I0, Δu; ndrange = Nu[α])
        end
        nothing
    end
    ap!
end




##################
#       PBC
##################
# Redefine the apply_bc_p! function in order to comply with Enzyme
using KernelAbstractions
using Enzyme
Enzyme.API.runtimeActivity!(true)

myapply_bc_p! = get_bc_p!(cache_p, setup) 



############# Test a similar thing for the BC on u

myapply_bc_u! = get_bc_u!(cache_F, setup)





##### Redefine applypressure!
u = random_field(setup, T(0))
p = rand(T,(n+2,n+2))
myapplypressure! = get_applypressure!(u, setup)
myapplypressure!(u, p)#, Δu)
IncompressibleNavierStokes.applypressure!(u, p, setup)

if run_test
    # Speed test
    @timed for i in 1:1000
        u = random_field(setup, T(0))
        p = rand(T,(n+2,n+2))
        IncompressibleNavierStokes.applypressure!(u, p, setup)
    end
    @timed for i in 1:1000
        u = random_field(setup, T(0))
        p = rand(T,(n+2,n+2))
        myapplypressure!(u, p)#, Δu)
    end

    # Compare with INS
    for i in 1:1000
        u = random_field(setup, T(0))
        p = rand(T,(n+2,n+2))
        u0 = copy.(u)
        IncompressibleNavierStokes.applypressure!(u, p, setup)
        myapplypressure!(u0, p)#, Δu)
        @assert u == u0
    end

    # Check if it is differentiable
    u = random_field(setup, T(0))
    p = rand(T,(n+2,n+2))
    du = Enzyme.make_zero(u)
    dp = Enzyme.make_zero(p)
    dΔu = Enzyme.make_zero(Δu)
    @timed Enzyme.autodiff(Enzyme.Reverse, myapplypressure!, Const, DuplicatedNoNeed(u, du), DuplicatedNoNeed(p, dp))
end


##################Vy
##################Vy
##################Vy
# And now I redefine the momentum! function
get_momentum!(F, u, temp, setup) = let
    (; grid, bodyforce, workgroupsize, Re) = setup
    (; dimension, Nu, Iu, Δ, Δu, A) = grid
    D = dimension()

    function get_convectiondiffusion!(B)
        e = Offset{D}()
        ν = 1 / Re
        (; Δ, Δu, A) = grid
        @kernel function cd!(F, u, ::Val{α}, ::Val{βrange}, I0, Δu, Δ, ν, A) where {α,βrange}
            I = @index(Global, Cartesian)
            I = I + I0
            KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
                #Δuαβ = α == β ? Δu[:,β] : Δ[:,β]
                Δuαβ = α == β ? Δu[β] : Δ[β]
                uαβ1 = (u[α][I-e(β)] + u[α][I]) / 2
                uαβ2 = (u[α][I] + u[α][I+e(β)]) / 2
                uβα1 =
                    A[β][α][2][I[α]-(α==β)] * u[β][I-e(β)] +
                    A[β][α][1][I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
                uβα2 = A[β][α][2][I[α]] * u[β][I] + A[β][α][1][I[α]+1] * u[β][I+e(α)]
                uαuβ1 = uαβ1 * uβα1
                uαuβ2 = uαβ2 * uβα2
                #∂βuα1 = (u[α][I] - u[α][I-e(β)]) / (β == α ? Δ[I[β],β] : Δu[I[β]-1,β])
                #∂βuα2 = (u[α][I+e(β)] - u[α][I]) / (β == α ? Δ[I[β]+1,β] : Δu[I[β],β])
                ∂βuα1 = (u[α][I] - u[α][I-e(β)]) / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
                ∂βuα2 = (u[α][I+e(β)] - u[α][I]) / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
                F[α][I] += (ν * (∂βuα2 - ∂βuα1) - (uαuβ2 - uαuβ1)) / Δuαβ[I[β]]
            end
        end
        function convdiff!(F, u)#, Δ, Δu)#, ν)#, A)
            for α = 1:D
                I0 = first(Iu[α])
                I0 -= oneunit(I0)
                cd!(B, workgroupsize)(F, u, Val(α), Val(1:D), I0, Δu, Δ, ν, A; ndrange = Nu[α])
            end
            nothing
        end
    end
    function get_bodyforce!(F, u, setup)
        @error "Not implemented"
    end
    convectiondiffusion! = get_convectiondiffusion!(get_backend(F[1]))
    bodyforce! = isnothing(bodyforce) ? (F,u,t)->nothing : get_bodyforce!(F, u, setup)
    gravity! = isnothing(temp) ? (F,u,t)->nothing : INS.gravity!(F, temp, setup)
    function momentum!(F, u, t)#, Δ, Δu)#, ν)#, A, t)
        for α = 1:D
            F[α] .= 0
        end
        convectiondiffusion!(F, u)#, Δ, Δu)#, ν)#, A)
        bodyforce!(F, u, t)
        gravity!(F, temp, setup)
        nothing
    end
end


(; Δ, Δu, A) = grid
ν = 1 / Re

u = random_field(setup, T(0))
F = random_field(setup, T(0))
my_f = get_momentum!(F, u, nothing, setup)
sΔ = stack(Δ)
sΔu = stack(Δu)
my_f(F, u, T(0))#, sΔ, sΔu)#, ν)#, A, T(0))

if run_test
    # Check if it is differentiable
    u = random_field(setup, T(0))
    F = random_field(setup, T(0))
    du = Enzyme.make_zero(u)
    dF = Enzyme.make_zero(F)
    dΔu = Enzyme.make_zero(Δu)
    dΔ = Enzyme.make_zero(Δ)
    dν = Enzyme.make_zero(ν)
    dA = Enzyme.make_zero(A)
    dsΔ = Enzyme.make_zero(sΔ)
    dsΔu = Enzyme.make_zero(sΔu)
#    @timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), Const(T(0)), DuplicatedNoNeed(sΔ,dsΔ), DuplicatedNoNeed(sΔu, dsΔu))
    @timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), Const(T(0)))

    @timed for i in 1:1000
        u = random_field(setup, T(0))
        F = random_field(setup, T(0))
        IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
    end
    @timed for i in 1:1000
        u = random_field(setup, T(0))
        F = random_field(setup, T(0))
        my_f(F, u, T(0))#, sΔ, sΔu)
    end

    # Check if the implementation is correct
    for i in 1:1000
        u = random_field(setup, T(0))
        F = random_field(setup, T(0))
        u0 = copy.(u)
        F0 = copy.(F)
        IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
        my_f(F0, u, T(0))#, sΔ, sΔu)
        @assert F == F0
    end
end