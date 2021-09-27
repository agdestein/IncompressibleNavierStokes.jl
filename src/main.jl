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
        constants_ke()
    end

    # Construct mesh
    create_mesh!(setup)

    # Boundary conditions
    create_boundary_conditions!(setup)

    # construct operators (matrices) which are time-independent
    build_operators!(setup)

    # Initialization of solution vectors
    V_start, p_start = create_intial_conditions(setup)

    # Boundary conditions
    set_bc_vectors!(setup, t)

    # Construct body force or immersed boundary method
    # The body force is called in the residual routines e.g. F.m
    # Steady force can be precomputed once:
    if !setup.force.force_unsteady
        setup.force.Fx, setup.force.Fy, _ = force(V_start, t, setup, false)
    end

    # Input checking
    check_input!(setup)

    # Choose between steady and unsteady
    if setup.case.steady
        # Steady
        if setup.case.visc == "keps"
            # Steady flow with k-epsilon model, 2nd order
            V, p = solve_steady_ke()
        elseif setup.case.visc == "laminar"
            if setup.discretization.order4
                # Steady flow with laminar viscosity model, 4th order
                V, p = solve_steady()
            else
                if setup.ibm.ibm
                    #Steady flow with laminar viscosity model and immersed boundary method, 2nd order
                    V, p = solve_steady_ibm()
                else
                    # Steady flow with laminar viscosity model, 2nd order
                    V, p = solve_steady()
                end
            end
        elseif setup.case.visc == "ML"
            # Steady flow with mixing length, 2nd order
            V, p = solve_steady()
        else
            error("wrong value for visc parameter")
        end
    else
        # Unsteady
        if setup.case.visc == "keps"
            # Unsteady flow with k-eps model, 2nd order
            V, p = solve_unsteady_ke()
        elseif setup.case.visc ∈ ["laminar", "qr", "LES", "ML"]
            if setup.rom.rom
                # Unsteady flow with reduced order model with $(setup.rom.M) modes
                V, p = solve_unsteady_rom()
            else
                # Unsteady flow with laminar or LES model
                V, p = solve_unsteady()
            end
        else
            error("wrong value for visc parameter")
        end
        println("simulated time: $t")
    end
    totaltime = Base.time() - startime
    println("total elapsed CPU time: $totaltime")

    # Post-processing
    post_processing()

    (; V, p, setup, totaltime)
end
