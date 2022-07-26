"""
    Operators()

Discrete operators.
"""
Base.@kwdef mutable struct Operators{T}
    Au_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Au_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Au_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Av_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aw_wz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Iu_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iv_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iw_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iu_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iv_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iw_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iu_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iv_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Iw_wz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    M::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    G::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Bup::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bvp::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Bwp::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Cux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cuy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cuz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cwx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cwy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Su_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Su_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Sv_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sv_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Sw_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sw_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sw_wz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sw_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Sw_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Dux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Duy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Duz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dvx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dvy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dvz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dwx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dwy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Dwz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Diff::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Wu_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Wu_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Wv_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Wv_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Ww_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Ww_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Cux_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cuy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvx_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Cvy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Auy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Auz_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Avx_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Avz_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Awx_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Awy_k::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    A::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

    Aν_ux::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_uy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_uz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_vx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_vy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_vz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_wx::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_wy::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    Aν_wz::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)

end
