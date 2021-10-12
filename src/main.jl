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
    V_start, p_start, t = create_initial_conditions(setup)

    # Boundary conditions
    set_bc_vectors!(setup, t)

    # Input checking
    solution = check_input!(setup, V_start, p_start, t)

    # Choose between steady and unsteady
    if setup.case.is_steady
        # Steady
        if setup.case.visc == "keps"
            # Steady flow with k-epsilon model, 2nd order
            V, p = solve_steady_ke!(solution, setup)
        elseif setup.case.visc == "laminar"
            if setup.discretization.order4
                # Steady flow with laminar viscosity model, 4th order
                V, p = solve_steady!(solution, setup)
            else
                if setup.ibm.ibm
                    # Steady flow with laminar viscosity model and immersed boundary method, 2nd order
                    V, p = solve_steady_ibm!(solution, setup)
                else
                    # Steady flow with laminar viscosity model, 2nd order
                    V, p = solve_steady!(solution, setup)
                end
            end
        elseif setup.case.visc == "ML"
            # Steady flow with mixing length, 2nd order
            V, p = solve_steady!(solution, setup)
        else
            error("wrong value for visc parameter")
        end
    else
        # Unsteady
        if setup.case.visc == "keps"
            # Unsteady flow with k-eps model, 2nd order
            V, p = solve_unsteady_ke!(solution, setup)
        elseif setup.case.visc ∈ ["laminar", "qr", "LES", "ML"]
            if setup.rom.use_rom
                # Unsteady flow with reduced order model with $(setup.rom.M) modes
                V, p = solve_unsteady_rom!(solution, setup)
            else
                # Unsteady flow with laminar or LES model
                V, p = solve_unsteady!(solution, setup)
            end
        else
            error("wrong value for visc parameter")
        end
    end
    totaltime = Base.time() - starttime

    # Post-processing
    # Postprocess(solution, setup)

    (; V, p, setup, totaltime, solution)
end
