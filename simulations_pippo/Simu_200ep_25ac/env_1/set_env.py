"""Resume and use the environment.
"""

import sys
import os
import shutil
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from  EnvBackward_step import EnvBackward_step
import numpy as np
from dolfin import Expression
import math

import os
cwd = os.getcwd()

#number of actuations for each episode
actuations_number = 25

def resume_env(plot=False,
               step=500,
               dump=500,
               remesh=False,
               random_start=False,
               single_run=False):
    # ---------------------------------------------------------------------------------
    # the configuration version number 1

    # duration of each simulations in second
    simulation_duration = 2.0
    # timestep
    dt=0.0005

    root = 'mesh/our_mesh'
    if(not os.path.exists('mesh')):
        os.mkdir('mesh')


    geometry_params = {'output': '.'.join([root, 'geo']),
                    'total_length': 3,
                    'frequency': 1,
                    'total_height' : 0.3,
                    'length_before_control' : 0.95,
                    'control_width' : 0.05,
                    'step_height' : 0.1,
                    'coarse_size': 0.1,
                    'coarse_distance': 0.5,
                    'box_size': 0.05,
                    #central point of control_width (used in the jet_bcs function)
                    'set_freq': 0,
                    'control_terms': ['Qs', 'frequencies'],
                    'tuning_parameters' : [4.0,1.0,0.0],
                    'clscale': 1,
                    'template': '../backward_facing_step.template_geo',
                    'remesh': remesh}

    if geometry_params['set_freq']:
        geometry_params['control_terms'] = ['Qs']

    def profile(mesh, degree):
        bot = mesh.coordinates().min(axis=0)[1]+0.1
        top = mesh.coordinates().max(axis=0)[1]
        print(bot, top)
        height = top - bot

        # streamline_velocity
        Umax = 1.5
        # inflow value, degree stands for the polynomial degree approximation
        return Expression(('-4*Umax*(x[1]-bot)*(x[1]-top)/height/height',
                        '0'), bot=bot, top=top, height=height, Umax=Umax, degree=degree)

    flow_params = {'mu': 1E-3,
                  'rho': 1,
                  'inflow_profile': profile}

    solver_params = {'dt': dt,
                    'solver_type': 'lu', # choose between lu(direct) and la_solve(iterative)
                    'preconditioner_step_1': 'default',
                    'preconditioner_step_2': 'amg',
                    'preconditioner_step_3': 'jacobi',
                    'la_solver_step_1': 'gmres',
                    'la_solver_step_2': 'gmres',
                    'la_solver_step_3': 'cg'}

    #initialization of the list containing the coordinates of the probes
    list_position_probes = []
    # we decided to collocate the probes in the more critical region for the recirculation area:
    # that is the area below the step.
    # It would be likely a good possible improvement to place some probes also in the upper area
    positions_probes_for_grid_x = np.linspace(1,2,27)[1:-1]
    positions_probes_for_grid_y = np.linspace(0,0.1,6)[1:-1]



    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))

    output_params = {'locations': list_position_probes,
                     'probe_type': 'velocity'
                     }

    optimization_params = {"num_steps_in_pressure_history": 1,
                        "min_value_jet_MFR": -1.e0,
                        "max_value_jet_MFR": 1.e0,
                        "smooth_control": (actuations_number/dt)*(0.1*0.0005/80), # 80/0.0005*...=0.1
                        "zero_net_Qs": False,
                        "random_start": random_start}

    inspection_params = {"plot": plot,
                        "step": step,
                        "dump": dump,
                        "range_pressure_plot": [-2.0, 1],
                        "show_all_at_reset": False,
                        "single_run":single_run
                        }

    reward_function = 'recirculation_area'

    verbose = 3

    number_steps_execution = int((simulation_duration/dt)/actuations_number) #2

    # Start with the initialization

    #Possibility of varying the value of n_iter (i.e. iterations for the baseline simulation)
    # according to the fact that there will be a remesh.
    if(remesh):
        n_iter = int(10.0 / dt)
        # n_iter = int(5.0 / dt)
        # n_iter = int(1.0 / dt)
        print("Make converge initial state for {} iterations".format(n_iter))
    else:
        n_iter = None


    #Processing the name of the simulation

    simu_name = 'Simu'

    if optimization_params["max_value_jet_MFR"] != 0.01:
        next_param = 'maxF' + str(optimization_params["max_value_jet_MFR"]) # [2:]
        simu_name = '_'.join([simu_name, next_param])
    if actuations_number != 80:
        next_param = 'NbAct' + str(actuations_number)
        simu_name = '_'.join([simu_name, next_param])
    next_param = ''
    if reward_function == 'recirculation_area':
        next_param = 'area'
    elif reward_function == 'max_recirculation_area':
        next_param = 'max_area'
    simu_name = '_'.join([simu_name, next_param])

    env_backward_step = EnvBackward_step(path_root=root,
                                    geometry_params=geometry_params,
                                    flow_params=flow_params,
                                    solver_params=solver_params,
                                    output_params=output_params,
                                    optimization_params=optimization_params,
                                    inspection_params=inspection_params,
                                    n_iter_make_ready=n_iter,
                                    verbose=verbose,
                                    reward_function=reward_function,
                                    number_steps_execution=number_steps_execution,
                                    simu_name = simu_name)

    return(env_backward_step)
