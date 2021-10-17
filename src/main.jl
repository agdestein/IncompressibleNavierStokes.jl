"""
This m-file contains the code for the 2D incompressible Navier-Stokes
equations using a Finite Volume Method and a pressure correction
method.
- horizontal numbering of volumes
- 2nd and 4th order spatial (central) discretization convection and diffusion
- general boundary conditions different fourth order BC ("verstappen") can be used by changing
addpath("spatial/boundaryconditions/proposed") to addpath("spatial/boundaryconditions/verstappen") check operator_convection_diffusion construction of Duy and Dvx
see readme.txt
Benjamin Sanderse, September 2018 - April 2019
"""
function main(setup)
    # Start timer
    starttime = Base.time()

    # Turbulence constants
    if setup.case.visc == "keps"
        constants_ke!(setup)
    end

    # Construct mesh
    create_mesh!(setup)

    # Boundary conditions
    create_boundary_conditions!(setup)

    # Construct operators (matrices) which are time-independent
    build_operators!(setup)

    # Initialization of solution vectors
    V₀, p₀, t₀ = create_initial_conditions(setup)

    # Input checking
    check_input!(setup, V₀, p₀, t₀)

    # Solve problem
    problem = setup.case.problem
    V, p = solve(problem, setup, V₀, p₀)

    # Measure total time
    totaltime = Base.time() - starttime

    # Post-processing
    # postprocess(setup, V, p, setup.time.t_end)

    (; V, p, totaltime)
end
