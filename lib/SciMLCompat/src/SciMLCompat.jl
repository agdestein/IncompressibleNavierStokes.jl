module SciMLCompat

import IncompressibleNavierStokes as INS
include("enzyme.jl")

"""
    create_right_hand_side(setup, psolver)

Create right hand side function f(u, p, t) compatible with
the OrdinaryDiffEq ODE solvers. Note that `u` has to be an array.
To convert the tuple `u = (ux, uy)` to an array, use `stack(u)`.
"""
create_right_hand_side(setup, psolver) = function right_hand_side(u, p, t)
    u = eachslice(u; dims = ndims(u))
    u = (u...,)
    u = INS.apply_bc_u(u, t, setup)
    F = INS.momentum(u, nothing, t, setup)
    F = INS.apply_bc_u(F, t, setup; dudt = true)
    PF = INS.project(F, setup; psolver)
    PF = INS.apply_bc_u(PF, t, setup; dudt = true)
    stack(PF)
end

"""
    create_right_hand_side_inplace(setup, psolver)

In place version of [`create_right_hand_side`](@ref).
"""
function create_right_hand_side_inplace(setup, psolver)
    (; x, N, dimension) = setup.grid
    D = dimension()
    F = ntuple(α -> similar(x[1], N), D)
    div = similar(x[1], N)
    p = similar(x[1], N)
    function right_hand_side!(dudt, u, params, t)
        u = eachslice(u; dims = ndims(u))
        INS.apply_bc_u!(u, t, setup)
        INS.momentum!(F, u, nothing, t, setup)
        INS.apply_bc_u!(F, t, setup; dudt = true)
        INS.project!(F, setup; psolver, div, p)
        INS.apply_bc_u!(F, t, setup; dudt = true)
        for α = 1:D
            dudt[ntuple(Returns(:), D)..., α] .= F[α]
        end
        dudt
    end
end

"""
    create_right_hand_side_enzyme(_backend, setup)

It defines the right hand side function for the Enzyme AD. 
To do so, it has to precompile and wrap all the intermediate operations 
as explained more in detail in enzyme.jl.
"""
function create_right_hand_side_enzyme(_backend, setup, T, n)
    e_bc_u! = _get_enz_bc_u!(_backend, setup);
    e_bc_p! = _get_enz_bc_p!(_backend, setup);
    e_momentum! = _get_enz_momentum!(_backend, nothing, setup);
    e_divergence! = _get_enz_div!(_backend, setup);
    e_psolve! = _get_enz_psolver!(setup);
    e_applypressure! = _get_enz_applypressure!(_backend, setup);
    Ω = setup.grid.Ω;
    
    N = n+2
    f=zeros(T, (N,N,2))
    div=zeros(T,(N,N))
    p=zeros(T,(N,N))
    ft=zeros(T,n*n+1)
    pt=zeros(T,n*n+1)

    function F_ip(du, u, param, t) 
        u_view = eachslice(u; dims = 3)
        F = eachslice(f; dims = 3)
        e_bc_u!(u_view)
        e_momentum!(F, u_view, t)
        e_bc_u!(F)
        e_divergence!(div, F, p)
        @. div *= Ω
        e_psolve!(p, div, ft, pt)
        e_bc_p!(p)
        e_applypressure!(F, p)
        e_bc_u!(F)
        du[:,:,1] .= F[1]
        du[:,:,2] .= F[2]
        nothing
    end;
end

export create_right_hand_side, create_right_hand_side_inplace, create_right_hand_side_enzyme, _get_enz_bc_u!, _get_enz_bc_p!, _get_enz_momentum!, _get_enz_div!, _get_enz_psolver!, _get_enz_applypressure!

end
