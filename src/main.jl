"""
    main(setup)

Build mesh and operators from `setup`, solve problem, and postprocess.
Return solution and timing.
"""
function main(setup)
    starttime = Base.time()

    # Construct mesh and discrete operators
    create_mesh!(setup)
    create_boundary_conditions!(setup)
    build_operators!(setup)

    # Initialization of solution vectors
    V₀, p₀, t₀ = create_initial_conditions(setup)

    check_input!(setup, V₀, p₀, t₀)

    # Solve problem
    problem = setup.case.problem
    V, p = solve(problem, setup, V₀, p₀)

    totaltime = Base.time() - starttime

    # postprocess(setup, V, p, setup.time.t_end)

    (; V, p, totaltime)
end
