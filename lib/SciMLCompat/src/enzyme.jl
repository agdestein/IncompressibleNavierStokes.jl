
""" 
This module wraps some of the functions of the IncompressibleNavierStokes module
in order to make them compatible with Enzyme AD.

There are also some tests to check if the redefined functions are faster than the original ones and if they are differentiable.
"""

using IncompressibleNavierStokes 
using SparseArrays
using LinearAlgebra
using KernelAbstractions
using Enzyme
# runtimeActivity is required to tell Enzyme that all temporary variables should be active
Enzyme.API.runtimeActivity!(true)

struct Offset{D} end
@inline (::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))

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
    _get_enz_momentum!(_backend, temp, setup)

This function is used to precompile the momentum function.
The bodyforce is not yet implemented, while the gravity 
is implemented in the IncompressibleNavierStokes module (but not tested here).
"""
function _get_enz_momentum!(_backend, temp, setup) 
    (; grid, bodyforce, workgroupsize, Re) = setup
    (; dimension, Nu, Iu, Δ, Δu, A) = grid
    D = dimension()

    function get_convectiondiffusion!()
        e = Offset{D}()
        ν = 1 / Re
        (; Δ, Δu, A) = grid
        @kernel function cd!(F, u, ::Val{α}, ::Val{βrange}, I0, Δu, Δ, ν, A) where {α,βrange}
            I = @index(Global, Cartesian)
            I = I + I0
            KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
                Δuαβ = α == β ? Δu[β] : Δ[β]
                uαβ1 = (u[α][I-e(β)] + u[α][I]) / 2
                uαβ2 = (u[α][I] + u[α][I+e(β)]) / 2
                uβα1 =
                    A[β][α][2][I[α]-(α==β)] * u[β][I-e(β)] +
                    A[β][α][1][I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
                uβα2 = A[β][α][2][I[α]] * u[β][I] + A[β][α][1][I[α]+1] * u[β][I+e(α)]
                uαuβ1 = uαβ1 * uβα1
                uαuβ2 = uαβ2 * uβα2
                ∂βuα1 = (u[α][I] - u[α][I-e(β)]) / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
                ∂βuα2 = (u[α][I+e(β)] - u[α][I]) / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
                F[α][I] += (ν * (∂βuα2 - ∂βuα1) - (uαuβ2 - uαuβ1)) / Δuαβ[I[β]]
            end
        end
        function convdiff!(F, u)
            for α = 1:D
                I0 = first(Iu[α])
                I0 -= oneunit(I0)
                cd!(_backend, workgroupsize)(F, u, Val(α), Val(1:D), I0, Δu, Δ, ν, A; ndrange = Nu[α])
            end
            nothing
        end
    end
    function get_bodyforce!(F, u, setup)
        @error "Not implemented"
    end
    convectiondiffusion! = get_convectiondiffusion!()
    bodyforce! = isnothing(bodyforce) ? (F,u,t)->nothing : (F,u,t)->get_bodyforce!(F, u, setup)
    gravity! = isnothing(temp) ? (F,temp,setup)->nothing : (F,temp,setup)->IncompressibleNavierStokes.gravity!(F, temp, setup)
    function momentum!(F, u, t)
        for α = 1:D
            F[α] .= 0
        end
        convectiondiffusion!(F, u)
        bodyforce!(F, u, t)
        gravity!(F, temp, setup)
        nothing
    end
end


"""
    _get_enz_div!(_backend, setup)

This function is used to precompile the function that computes the divergence in place.
In particular, it precompiles Δ inside the function such that Enzyme does not try to Dual() it and fail because it is a tuple.
Also, the automatic differentiations requires a cache d to be preallocated and passed as an argument, on top of div.
"""

function _get_enz_div!(_backend, setup) 
    (; grid, workgroupsize) = setup
    (; dimension, Δ, Ip, Np) = grid
    D = dimension()
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
    _cache_psolver(grid, setup)

This function is used to precompile the parameters of the psolver function.
Its implementation is similar to the one in IncompressibleNavierStokes.
"""
function _cache_psolver(::Array, setup)
    (; grid, boundary_conditions) = setup
    (; x, Np, Ip) = grid
    T = eltype(x[1])
    L = IncompressibleNavierStokes.laplacian_mat(setup)
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
    _get_enz_psolver!(setup)

This function is used to precompile the psolver function.
In particular, it has to wrap fact, viewrange, and Ip such that Enzyme does not have to allocate memory for them.
"""

function _get_enz_psolver!(setup)
    fact, viewrange, Ip = _cache_psolver(setup.grid.x[1], setup)
    function psolver(p, f, ftemp, ptemp)
        copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
        ptemp .= fact \ ftemp
        copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
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
end