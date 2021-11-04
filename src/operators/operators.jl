# Discrete operators 
Base.@kwdef mutable struct Operators{T}
    order4::Bool = false                     # Use 4th order in space (otherwise 2nd order)
    α::T = 81                                # Richardson extrapolation factor = 3^4
    β::T = 9 // 8                            # Interpolation factor

    Au_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Au_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Au_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Au_ux_bc::NamedTuple = (;)
    Au_uy_bc::NamedTuple = (;)
    Au_uz_bc::NamedTuple = (;)
    Av_vx_bc::NamedTuple = (;)
    Av_vy_bc::NamedTuple = (;)
    Av_vz_bc::NamedTuple = (;)
    Aw_wx_bc::NamedTuple = (;)
    Aw_wy_bc::NamedTuple = (;)
    Aw_wz_bc::NamedTuple = (;)

    Iu_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iv_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iu_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iv_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Iu_ux_bc::NamedTuple = (;)
    Iv_vy_bc::NamedTuple = (;)
    Iv_uy_bc_lr::NamedTuple = (;)
    Iv_uy_bc_lu::NamedTuple = (;)
    Iu_vx_bc_lr::NamedTuple = (;)
    Iu_vx_bc_lu::NamedTuple = (;)

    M::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Mx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    My::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Mx_bc::NamedTuple = (;)
    My_bc::NamedTuple = (;)

    G::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Gx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Gy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Bup::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvp::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Cux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cuy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Su_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    # Su_vy never defined
    # Sv_ux never defined
    Sv_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Su_ux_bc::NamedTuple = (;)
    Su_uy_bc::NamedTuple = (;)
    Sv_vx_bc::NamedTuple = (;)
    Sv_vy_bc::NamedTuple = (;)

    Su_vx_bc_lr::NamedTuple = (;)
    Su_vx_bc_lu::NamedTuple = (;)
    Sv_uy_bc_lr::NamedTuple = (;)
    Sv_uy_bc_lu::NamedTuple = (;)

    Dux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Duy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dvx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Diff::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Wv_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Wu_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    yM::Vector{T} = T[]
    y_p::Vector{T} = T[]
	
    yAu_ux::Vector{T} = T[]
    yAu_uy::Vector{T} = T[]
    yAv_vx::Vector{T} = T[]
    yAv_vy::Vector{T} = T[]
    
	yDiff::Vector{T} = T[]
    
	yIu_ux::Vector{T} = T[]
    yIv_uy::Vector{T} = T[]
    yIu_vx::Vector{T} = T[]
    yIv_vy::Vector{T} = T[]

    ySu_ux::Vector{T} = T[]
    ySu_uy::Vector{T} = T[]
    ySu_vx::Vector{T} = T[]
    ySv_uy::Vector{T} = T[]
    ySv_vx::Vector{T} = T[]
    ySv_vy::Vector{T} = T[]

    Cux_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cuy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvx_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Cux_k_bc::NamedTuple = (;)
    Cuy_k_bc::NamedTuple = (;)
    Cvx_k_bc::NamedTuple = (;)
    Cvy_k_bc::NamedTuple = (;)

    Auy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Avx_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Auy_k_bc::NamedTuple = (;)
    Avx_k_bc::NamedTuple = (;)

    A::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    ydM::Vector{T} = T[]
    ypx::Vector{T} = T[]
    ypy::Vector{T} = T[]

    Aν_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_ux_bc::NamedTuple = (;)
    Aν_vy_bc::NamedTuple = (;)
    Aν_uy_bc_lr::NamedTuple = (;)
    Aν_uy_bc_lu::NamedTuple = (;)
    Aν_vx_bc_lr::NamedTuple = (;)
    Aν_vx_bc_lu::NamedTuple = (;)

    yAν_ux::Vector{T} = T[]
    yAν_uy::Vector{T} = T[]
    yAν_vx::Vector{T} = T[]
    yAν_vy::Vector{T} = T[]

    yCux_k::Vector{T} = T[]
    yCuy_k::Vector{T} = T[]
    yCvx_k::Vector{T} = T[]
    yCvy_k::Vector{T} = T[]
    yAuy_k::Vector{T} = T[]
    yAvx_k::Vector{T} = T[]
end
