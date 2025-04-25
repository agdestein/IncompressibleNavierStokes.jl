"""
    IncompressibleNavierStokesAMGX extension

Introduces fast CG method for Poisson solver using AMGX.
"""
module IncompressibleNavierStokesAMGXExt

using AMGX
using IncompressibleNavierStokes
using IncompressibleNavierStokes: PressureBC, laplacian_mat
using SparseArrays
using AMGX.CUDA

function IncompressibleNavierStokes.amgx_setup()
    AMGX.initialize()
    config = AMGX.Config(
        Dict(
            "config_version" => "2",
            "solver" => "CG",
            "cg:tolerance" => 0.0001,
            "cg:max_iters" => 100,
        ),
    )
    # config = AMGX.Config(Dict(
    #      "config_version" => "2",
    #      "solver" => "PCGF",
    #      "pcgf:tolerance" => 0.0001,
    #      "pcgf:max_iters" => 100,
    #      "preconditioner(amg_solver)" => "AMG",
    #      "amg_solver:algorithm" => "CLASSICAL",
    #      "amg_solver:max_iters" => 1,
    # ))
    resources = AMGX.Resources(config);
    rhs = AMGX.AMGXVector(resources, AMGX.dFFI)
    matrix = AMGX.AMGXMatrix(resources, AMGX.dFFI)
    AMGXsolver = AMGX.Solver(resources, AMGX.dFFI, config)
    solution = AMGX.AMGXVector(resources, AMGX.dFFI)
    stuff = (; config, resources, rhs, matrix, AMGXsolver, solution)
end

function IncompressibleNavierStokes.close_amgx(stuff)
    # need to finalize:
    close(stuff.rhs)
    close(stuff.matrix)
    close(stuff.solution)
    close(stuff.AMGXsolver)
    close(stuff.resources)
    close(stuff.config)
    AMGX.finalize()
end

function IncompressibleNavierStokes.psolver_cg_AMGX(setup; stuff, kwargs...)
    (; grid, boundary_conditions, backend) = setup
    (; x, Np, Ip) = grid
    T = eltype(x[1])
    L = laplacian_mat(setup)
    isdefinite = true
    #any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
    if isdefinite
        # No extra DOF
        ftemp = fill!(similar(x[1], prod(Np)), 0)
        ptemp = fill!(similar(x[1], prod(Np)), 0)
        viewrange = (:)
    else
        # With extra DOF
        ftemp = fill!(similar(x[1], prod(Np) + 1), 0)
        ptemp = fill!(similar(x[1], prod(Np) + 1), 0)
        e = fill(T(1), prod(Np))
        L = [L e; e' 0]
        viewrange = 1:prod(Np)
    end
    L = CUDA.CUSPARSE.CuSparseMatrixCSR(L)

    AMGX.upload!(
        stuff.matrix,
        ((L.rowPtr) .- Int32(1)), # row_ptrs
        ((L.colVal) .- Int32(1)), # col_indices
        (L.nzVal), # data
    )

    AMGX.setup!(stuff.AMGXsolver, stuff.matrix)

    function psolve!(p)
        copyto!(view(ftemp, viewrange), view(view(p, Ip), :))

        AMGX.upload!(stuff.rhs, ftemp) # Upload right-hand side
        AMGX.set_zero!(stuff.solution, size(ftemp, 1))
        AMGX.solve!(stuff.solution, stuff.AMGXsolver, stuff.rhs)
        AMGX.copy!(ptemp, stuff.solution)

        # cg!(ptemp, L, ftemp; kwargs...)
        copyto!(view(view(p, Ip), :), view(ptemp, viewrange))
        p
    end
end

end
