"""
    Grid{T,N}()

Nonuniform cartesian grid of dimension `N` and floating point type `T`.
"""
Base.@kwdef struct Grid{T,N}
    Nx::Int = 10                             # Number of x-volumes
    Ny::Int = 10                             # Number of y-volumes
    Nz::Int = 1                              # Number of z-volumes (if any)
    xlims::Tuple{T,T} = (0, 1)               # Horizontal limits (left, right)
    ylims::Tuple{T,T} = (0, 1)               # Vertical limits (bottom, top)
    zlims::Tuple{T,T} = (0, 1)               # Depth limits (back, front)

    x::Vector{T} = T[]                       # Vector of x-points
    y::Vector{T} = T[]                       # Vector of y-points
    z::Vector{T} = T[]                       # Vector of z-points
    xp::Vector{T} = T[]
    yp::Vector{T} = T[]
    zp::Vector{T} = T[]

    # Number of pressure points in each dimension
    Npx::Int = 0
    Npy::Int = 0
    Npz::Int = 0

    Nux_in::Int = 0
    Nux_b::Int = 0
    Nux_t::Int = 0
    Nuy_in::Int = 0
    Nuy_b::Int = 0
    Nuy_t::Int = 0
    Nuz_in::Int = 0
    Nuz_b::Int = 0
    Nuz_t::Int = 0

    Nvx_in::Int = 0
    Nvx_b::Int = 0
    Nvx_t::Int = 0
    Nvy_in::Int = 0
    Nvy_b::Int = 0
    Nvy_t::Int = 0
    Nvz_in::Int = 0
    Nvz_b::Int = 0
    Nvz_t::Int = 0

    Nwx_in::Int = 0
    Nwx_b::Int = 0
    Nwx_t::Int = 0
    Nwy_in::Int = 0
    Nwy_b::Int = 0
    Nwy_t::Int = 0
    Nwz_in::Int = 0
    Nwz_b::Int = 0
    Nwz_t::Int = 0

    # Number of points in solution vector
    Nu::Int = 0
    Nv::Int = 0
    Nw::Int = 0
    NV::Int = 0
    Np::Int = 0

    # Operator mesh?
    Ωp::Vector{T} = T[]
    Ω::Vector{T} = T[]
    Ω⁻¹::Vector{T} = T[]

    # For order4
    Ωux::Vector{T} = T[]
    Ωvx::Vector{T} = T[]
    Ωwx::Vector{T} = T[]
    Ωuy::Vector{T} = T[]
    Ωvy::Vector{T} = T[]
    Ωwy::Vector{T} = T[]
    Ωuz::Vector{T} = T[]
    Ωvz::Vector{T} = T[]
    Ωwz::Vector{T} = T[]

    hx::Vector{T} = T[]
    hy::Vector{T} = T[]
    hz::Vector{T} = T[]
    hxi::Vector{T} = T[]
    hyi::Vector{T} = T[]
    hzi::Vector{T} = T[]
    hxd::Vector{T} = T[]
    hyd::Vector{T} = T[]
    hzd::Vector{T} = T[]
    gx::Vector{T} = T[]
    gy::Vector{T} = T[]
    gz::Vector{T} = T[]
    gxi::Vector{T} = T[]
    gyi::Vector{T} = T[]
    gzi::Vector{T} = T[]
    gxd::Vector{T} = T[]
    gyd::Vector{T} = T[]
    gzd::Vector{T} = T[]

    Buvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Buwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bwux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bwvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Bkux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bkvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bkwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    xin::Vector{T} = T[]
    yin::Vector{T} = T[]
    zin::Vector{T} = T[]

    # Separate grids for u, v, and p
    xu::Array{T,N} = zeros(T, fill(0, N)...)
    xv::Array{T,N} = zeros(T, fill(0, N)...)
    xw::Array{T,N} = zeros(T, fill(0, N)...)
    yu::Array{T,N} = zeros(T, fill(0, N)...)
    yv::Array{T,N} = zeros(T, fill(0, N)...)
    yw::Array{T,N} = zeros(T, fill(0, N)...)
    zu::Array{T,N} = zeros(T, fill(0, N)...)
    zv::Array{T,N} = zeros(T, fill(0, N)...)
    zw::Array{T,N} = zeros(T, fill(0, N)...)
    xpp::Array{T,N} = zeros(T, fill(0, N)...)
    ypp::Array{T,N} = zeros(T, fill(0, N)...)
    zpp::Array{T,N} = zeros(T, fill(0, N)...)

    # Ranges
    indu::UnitRange{Int} = 0:0
    indv::UnitRange{Int} = 0:0
    indw::UnitRange{Int} = 0:0
    indV::UnitRange{Int} = 0:0
    indp::UnitRange{Int} = 0:0
end

