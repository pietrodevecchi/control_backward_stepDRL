from tensorforce import TensorforceError
from tensorforce.environments import Environment

import sys
import os
cwd = os.getcwd()
# called into the environment folders and needs to copy data from simulation folder
sys.path.append(cwd + "/../Simulation/")

from dolfin import Expression, File, plot
from probes import PressureProbeValues, VelocityProbeValues, TotalrecirculationArea
from generate_msh import generate_mesh
from cfd_solver import FlowSolver
from msh_convert import convert
from dolfin import *

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import math
import csv
import shutil

# function to define inflow profile look into flow solver to find more comments
def constant_profile(mesh, degree): 
    '''
    Time independent inflow profile.
    '''
    bot = mesh.coordinates().min(axis=0)[1]+0.1
    top = mesh.coordinates().max(axis=0)[1]

    H = top - bot

    Um = 1.5

    return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                       '0'), bot=bot, top=top, H=H, Um=Um, degree=degree, time=0)



class RingBuffer():

    "A 1D ring buffer using numpy arrays to keep track of data with a maximum length"
   
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer in case you exceed the length you override restarting from the beginning"
        x_index = (self.index + np.arange(x.size)) % self.data.size      
                                                                      
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer (all of it)"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]




class EnvBackward_step(Environment):
    """
    Inherits from tensorflow class Environment and is adapted to our
    deep reinforcement learning task, overrides the principal functions
    to handle our cfd framework
    """

    def __init__(self, path_root, geometry_params, flow_params, solver_params, output_params,
                 optimization_params, inspection_params, n_iter_make_ready=None, verbose=0, size_history=2000,
                 reward_function='plain_drag', size_time_state=50, number_steps_execution=1, simu_name="Simu", 
                 ):
        
        # initialize parameters passed from env.py

        self.a1 = geometry_params['tuning_parameters'][0]
        self.a2 = geometry_params['tuning_parameters'][1]
        self.b = geometry_params['tuning_parameters'][2]
        self.control_width = geometry_params['control_width']


        # path of of mesh files in particular for remesh (creating baseline)
        self.path_root = path_root

        # cfd params: density viscosity and inflow profile
        self.flow_params = flow_params

        # mesh parameters (remesh and template included)
        self.geometry_params = geometry_params

        # time step and type of solver
        self.solver_params = solver_params

        # probes parameters
        self.output_params = output_params

        # for agent and actuation choices
        self.optimization_params = optimization_params

        # to keep track of processes
        self.inspection_params = inspection_params

        # level of terminal printing
        self.verbose = verbose

        # number of iteration for baseline simulation
        self.n_iter_make_ready = n_iter_make_ready

        # length of our buffers to store data for learning
        self.size_history = size_history          
  
        self.reward_function = reward_function

        self.size_time_state = size_time_state

        self.number_steps_execution = number_steps_execution

        self.simu_name = simu_name


        # reastart from last episode reading a file containing
        # checkpoint if present, otherwise start from episode 0
        name="output.csv"
        last_row = None
        if(os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, 'r') as f:
                for row in reversed(list(csv.reader(f, delimiter=";", lineterminator="\n"))):    
                    last_row = row                                                              
                    break 
        if(not last_row is None):
            self.episode_number = int(last_row[0])                                                # vedi qua
            self.last_episode_number = int(last_row[0])
        else:
            self.last_episode_number = 0
            self.episode_number = 0

        self.episode_areas = np.array([])

        self.initialized_visualization = False

        self.start_class()                               


#------------------------------------------------------------------- Start Class --------------------------------------------------------
    def start_class(self):

        # step of the episode
        self.solver_step = 0

        self.initialized_output = False

        self.resetted_number_probes = False

        self.area_probe = None

        # dictionary with RingBuffer objects to keep track of all our data
        self.history_parameters = {}

        # initialize buffer for action terms (both for amplitude and frequency)
        for contr_term in self.geometry_params['control_terms']:                          
            self.history_parameters["control_for_{}".format(contr_term)] = RingBuffer(self.size_history)   
        

        # dimension of action space for agent
        self.history_parameters["number_of_control_terms"] = len(self.geometry_params["control_terms"]) 
        

   
        # creating buffers for every probe named after its location (pressure or velocity probes can be used)
        for crrt_probe in range(len(self.output_params["locations"])):
            if self.output_params["probe_type"] == 'pressure':
                self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(self.size_history)
            elif self.output_params["probe_type"] == 'velocity':
                self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(self.size_history)
                self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(self.size_history)

        # number of probes and so dimension of the state space for pressure, half of it for velocity
        self.history_parameters["number_of_probes"] = len(self.output_params["locations"])

       # buffer initialized to store rewards
        self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

        # extracting mesh data data looking for files named path_root
        h5_file = '.'.join([self.path_root, 'h5'])
        msh_file = '.'.join([self.path_root, 'msh'])
        self.geometry_params['mesh'] = h5_file

 
        # if remesh is True, mesh is generated from template and converted to h5 file

        if self.geometry_params['remesh']:

            if self.verbose > 0: 
                print("Remesh")


            generate_mesh(self.geometry_params, template=self.geometry_params['template'])

            if self.verbose > 0:
                print("generate_mesh done!")

            print(msh_file)
            assert os.path.exists(msh_file)

            convert(msh_file, h5_file)
            assert os.path.exists(h5_file)



        if self.n_iter_make_ready is None:                        
            if self.verbose > 0:
                print("Load initial flow")

            # initial conditions loaded from xdmf files
            self.flow_params['u_init'] = 'mesh/u_init.xdmf'
            self.flow_params['p_init'] = 'mesh/p_init.xdmf'

            if self.verbose > 0:
                print("Load buffer history")
            
            # recover hystory parameters from pkl file
            with open('mesh/dict_history_parameters.pkl', 'rb') as f:  
                self.history_parameters = pickle.load(f)

            # initialize by hand parameters that were not present in the file
            if not "number_of_probes" in self.history_parameters:
                self.history_parameters["number_of_probes"] = 0
            if not "number_of_control_terms" in self.history_parameters:
                self.history_parameters["number_of_control_terms"] = len(self.geometry_params["control_terms"])
            if not "recirc_area" in self.history_parameters:
                self.history_parameters["recirc_area"] = RingBuffer(self.size_history)
            
            # if not the same number of probes, reset them one by one
            if not self.history_parameters["number_of_probes"] == len(self.output_params["locations"]):
                for crrt_probe in range(len(self.output_params["locations"])):
                    if self.output_params["probe_type"] == 'pressure':
                        self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(self.size_history)
                    elif self.output_params["probe_type"] == 'velocity':
                        self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(self.size_history)
                        self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(self.size_history)

                self.history_parameters["number_of_probes"] = len(self.output_params["locations"])

                self.resetted_number_probes = True

        
        # create the flow simulation object
        self.flow = FlowSolver(self.flow_params, self.geometry_params, self.solver_params)


        # ------------------------------------------------------------------------
        # once cfd object is created, probes and training parameters have to be set
        
        # Setup probes
        if self.output_params["probe_type"] == 'pressure':
            self.ann_probes = PressureProbeValues(self.flow, self.output_params['locations'])
        elif self.output_params["probe_type"] == 'velocity':
            self.ann_probes = VelocityProbeValues(self.flow, self.output_params['locations'])      

        # ------------------------------------------------------------------------
        
        # No flux from jets for starting
        if self.geometry_params['set_freq']:
            self.Qs= np.zeros(1)
            self.frequencies= np.zeros(1)
            self.control_evolution=np.zeros(1)
        else:
            [self.Qs, self.frequencies,self.control_evolution] = np.zeros(len(self.geometry_params['control_terms']))
    
        if self.geometry_params['set_control']:
            self.Qs= np.zeros(1)
            self.frequencies= np.zeros(1)
            self.control_evolution=np.zeros(1)
        else:
            [self.Qs, self.frequencies,self.control_evolution] = np.zeros(len(self.geometry_params['control_terms']))

        self.action = np.zeros(len(self.geometry_params['control_terms']))

        # ------------------------------------------------------------------------
        
        # prepare parameters for plotting functions
        self.compute_positions_for_plotting()   

        # ------------------------------------------------------------------------
       
        # implementation of a simulation without control
        # used to run the baseline simulation called by make_mesh.py in
        # simulation_base together with remesh
        if self.n_iter_make_ready is not None:

            # initialize solution
            if self.geometry_params['set_freq']:
                self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
            else:
                self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies,self.control_evolution])  
            if self.geometry_params['set_control']:
                self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
            else:
                self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies,self.control_evolution])        
            path=''

            # dump set the number of steps after which we save one
            if "dump" in self.inspection_params:             
                path = 'results/area_out.pvd'

            
            self.area_probe = TotalrecirculationArea(self.u_, 0, store_path=path) 

            if self.verbose > 0:
                print("Compute initial flow")
            
            # simulation with n_iter_make_readsy steps and no control
            for _ in range(self.n_iter_make_ready):
                if self.geometry_params['set_freq']:
                    self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
                else:
                    self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies,self.control_evolution])
                if self.geometry_params['set_control']:
                    self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
                else:
                    self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies,self.control_evolution])    
                # compute probe values (state)
                self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
                
                # compute percentage of domain with negative velocity (reward)
                self.recirc_area = self.area_probe.sample(self.u_, self.p_)

                # save data
                self.write_history_parameters()

                # displaying information that has to do with the solver itself
                self.visual_inspection()

                # print information
                self.output_data()

                self.solver_step += 1

        
            # convert the mesh to h5 file
            encoding = XDMFFile.Encoding.HDF5
            mesh = convert(msh_file, h5_file)
            comm = mesh.mpi_comm()

            # save field data as initial condition for control latee simulations
            XDMFFile(comm, 'mesh/u_init.xdmf').write_checkpoint(self.u_, 'u0', 0, encoding)
            XDMFFile(comm, 'mesh/p_init.xdmf').write_checkpoint(self.p_, 'p0', 0, encoding)

            # save buffer dict (the one opened above to retrieve infos if already present, is here initialized)
            with open('mesh/dict_history_parameters.pkl', 'wb') as f:
                pickle.dump(self.history_parameters, f, pickle.HIGHEST_PROTOCOL)

        # ----------------------------------------------------------------------
        
        
        # if reading from disk, we call again the situation of 
        # reading data from a checkpoint, the one called during training, but now we have created 
        # cfd solver object and probes objects so we can initialize missing part 

        if self.n_iter_make_ready is None:

            # iteration to initialize solution variables
            if self.geometry_params['set_freq']:
                self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
            else:
                self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies,self.control_evolution])
            if self.geometry_params['set_control']:
                self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
            else:
                self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies,self.control_evolution])    
            path=''
        
            # as before file to store data every dump time steps
            if "dump" in self.inspection_params:
                path = 'results/area_out.pvd'
            self.area_probe = TotalrecirculationArea(self.u_, 0, store_path=path)

            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()

            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            self.write_history_parameters()
            self.visual_inspection()
            self.output_data()

        # ----------------------------------------------------------------------
        
        # if necessary, fill the probes buffer (done after remesh)
        if self.resetted_number_probes:
            for _ in range(self.size_history):
                self.execute()

        # ----------------------------------------------------------------------
        
        # now it's all set

        self.ready_to_use = True

  
  
  
# ------------------------------------------------------------------------------------- End of Start_class  ---------------------------------------------------------------
  
  
  

    def write_history_parameters(self):
        
        # save actuation values
        self.history_parameters["control_for_Qs"].extend(self.Qs)
        self.history_parameters["control_for_frequencies"].extend(self.frequencies)
        self.history_parameters["control_for_control_evolution"].extend(self.control_evolution)
        # save probes values
        if self.output_params["probe_type"] == 'pressure':
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}".format(crrt_probe)].extend(self.probes_values[crrt_probe])
        elif self.output_params["probe_type"] == 'velocity':
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}_u".format(crrt_probe)].extend(self.probes_values[2 * crrt_probe])
                self.history_parameters["probe_{}_v".format(crrt_probe)].extend(self.probes_values[2 * crrt_probe + 1])

        # save rewarda value
        self.history_parameters["recirc_area"].extend(np.array(self.recirc_area))
        
        
        

    def compute_positions_for_plotting(self):

        # where the probes are
        self.list_positions_probes_x = []
        self.list_positions_probes_y = []

        total_number_of_probes = len(self.output_params['locations'])

        #printiv(total_number_of_probes)

        # get the positions
        for crrt_probe in self.output_params['locations']:    
            # if self.verbose > 2:
            #     print(crrt_probe)

            self.list_positions_probes_x.append(crrt_probe[0])
            self.list_positions_probes_y.append(crrt_probe[1])    

 
        x_pos = self.geometry_params['length_before_control'] + self.geometry_params['control_width']
        y_pos = self.geometry_params['step_height']
        self.positions_jet_x = []
        self.positions_jet_y = []
        self.positions_jet_x.append(x_pos)
        self.positions_jet_y.append(y_pos)

            

 #-------------------------------------------------------------------------------------------------------------------------


    def show_flow(self):
        plt.figure()
        plot(self.u_)
        plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
        plt.scatter(self.positions_jet_x, self.positions_jet_y, c='r', marker='o')
        plt.xlim([0, self.geometry_params['total_length']])
        plt.ylim([0, self.geometry_params['total_height']])
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.show()

        plt.figure()
        p = plot(self.p_)
        cb = plt.colorbar(p, fraction=0.1, shrink=0.3)
        plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
        plt.scatter(self.positions_jet_x, self.positions_jet_y, c='r', marker='o')
        plt.xlim([0, self.geometry_params['total_length']])
        plt.ylim([0, self.geometry_params['total_height']])
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.tight_layout()
        cb.set_label("P")
        plt.show()
        
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def show_control(self):
        plt.figure()

        linestyles = ['-', '--', ':', '-.']

        for contr_term in self.geometry_params['control_terms']:
            contr_term_data = self.history_parameters["control_for_{}".format(contr_term)].get()
            plt.plot(contr_term_data, label="jet {}".format(contr_term), linestyle=linestyles[contr_term], linewidth=1.5)
        plt.legend(loc=2)
        plt.ylabel("control")
        plt.xlabel("actuation step")
        plt.tight_layout()
        plt.pause(1.0)
        plt.savefig("saved_figures/control_episode_{}.pdf".format(self.episode_number))
        plt.show()
        plt.pause(2.0)


#------------------------------------------------------------------------------------------------------------------------------------------------------
        

    def visual_inspection(self):
        total_number_subplots = 4
        crrt_subplot = 1

        if(not self.initialized_visualization and self.inspection_params["plot"] != False):
            plt.ion()
            plt.subplots(total_number_subplots, 1)

            self.initialized_visualization = True

        if("plot" in self.inspection_params and self.inspection_params["plot"] != False):
            # plot each modulo_base time steps
            modulo_base = self.inspection_params["plot"]

            if self.solver_step % modulo_base == 0:

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plot(self.u_)
                plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
                plt.scatter(self.positions_jet_x, self.positions_jet_y, c='r', marker='o')
                plt.xlim([0, self.geometry_params['total_length']])
                plt.ylim([0, self.geometry_params['total_height']])
                plt.ylabel("V")
                crrt_subplot += 1

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plot(self.p_)
                plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
                plt.scatter(self.positions_jet_x, self.positions_jet_y, c='r', marker='o')
                plt.xlim([0, self.geometry_params['total_length']])
                plt.ylim([0, self.geometry_params['total_height']])
                plt.ylabel("P")
                crrt_subplot += 1

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plt.cla()
                for contr_term in self.geometry_params['control_terms']:
                    contr_term_data = self.history_parameters["control_for_{}".format(contr_term)].get()
                    plt.plot(contr_term_data, label="jet {}".format(contr_term))
                plt.legend(loc=6)
                plt.ylabel("M.F.R.")
                crrt_subplot += 1


                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plt.cla()
                crrt_area = self.history_parameters["recirc_area"].get()
                plt.plot(crrt_area)
                plt.ylabel("RecArea")
                plt.xlabel("buffer steps")
                plt.ylim([0, 0.03])

                # plt.tight_layout()
                plt.tight_layout(pad=0, w_pad=0, h_pad=-0.5)
                plt.draw()
                plt.pause(0.5)

        if self.solver_step % self.inspection_params["dump"] == 0 and self.inspection_params["dump"] < 10000:

            print("%s | Ep N: %4d, step: %4d, Rec Area: %.4f"%(self.simu_name,
            self.episode_number,
            self.solver_step,
            self.history_parameters["recirc_area"].get()[-1]))
            
            name = "debug.csv"
            if(not os.path.exists("saved_models")):
                os.mkdir("saved_models")
            if(not os.path.exists("saved_models/"+name)):
                with open("saved_models/"+name, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Name", "Episode", "Step", "RecircArea"])
                    spam_writer.writerow([self.simu_name,
                                          self.episode_number,
                                          self.solver_step,
                                          self.history_parameters["recirc_area"].get()[-1]])
            else:
                with open("saved_models/"+name, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.simu_name,
                                          self.episode_number,
                                          self.solver_step,
                                          self.history_parameters["recirc_area"].get()[-1]])
                

        if("single_run" in self.inspection_params and self.inspection_params["single_run"] == True):
            # if ("dump" in self.inspection_params and self.inspection_params["dump"] > 10000):
                self.sing_run_output()
            
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            

    def sing_run_output(self):
        name = "test_strategy.csv"
        if(not os.path.exists("saved_models")):
            os.mkdir("saved_models")
        if(not os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, "w") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Name", "Step", "RecircArea"] + ["Control_action_n" + str(v) for v in range(len([self.Qs,self.frequencies,self.control_evolution]))])
                spam_writer.writerow([self.simu_name, self.solver_step,
                                      self.history_parameters["recirc_area"].get()[-1]] + [str(v) for v in [self.Qs,self.frequencies,self.control_evolution]])
        else:
            with open("saved_models/"+name, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.simu_name, self.solver_step,
                                      self.history_parameters["recirc_area"].get()[-1]] + [str(v) for v in [self.Qs,self.frequencies,self.control_evolution]])
        return
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

    def output_data(self):
        if "step" in self.inspection_params:
            modulo_base = self.inspection_params["step"]

            if self.solver_step % modulo_base == 0:
                if self.verbose > 0:
                    print("Solver step: {}".format(self.solver_step))
                    print("Amplitude_value: {}".format(self.a1*self.Qs))
                    print("Frequency: {}".format(self.a2*self.frequencies))
                    print("Control_evolution: {}".format(self.control_width + self.b*self.control_evolution))
                    # print(self.probes_values)
                    print("Recirc area: {}".format(self.recirc_area))
                    pass

        if "dump" in self.inspection_params and self.inspection_params["dump"] < 10000:
            modulo_base = self.inspection_params["dump"]

            self.episode_areas = np.append(self.episode_areas, [self.history_parameters["recirc_area"].get()[-1]])

            if(self.last_episode_number != self.episode_number and "single_run" in self.inspection_params and self.inspection_params["single_run"] == False):
                self.last_episode_number = self.episode_number
                avg_area = np.average(self.episode_areas[len(self.episode_areas)//2:])
                name = "output.csv"
                if(not os.path.exists("saved_models")):
                    os.mkdir("saved_models")
                if(not os.path.exists("saved_models/"+name)):
                    with open("saved_models/"+name, "w") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                        spam_writer.writerow(["Episode", "AvgRecircArea"])
                        spam_writer.writerow([self.last_episode_number,
                                              avg_area])
                else:
                    with open("saved_models/"+name, "a") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                        spam_writer.writerow([self.last_episode_number,
                                              avg_area])
                self.episode_areas = np.array([])

                if(os.path.exists("saved_models/output.csv")):
                    if(not os.path.exists("best_model")):
                        shutil.copytree("saved_models", "best_model")

                    else :
                        with open("saved_models/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            last_iter = lastrow[1]

                        with open("best_model/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            best_iter = lastrow[1]

                        if float(best_iter) > float(last_iter):
                            print("best_model updated")
                            if(os.path.exists("best_model")):
                                shutil.rmtree("best_model")
                            shutil.copytree("saved_models", "best_model")

            if self.solver_step % modulo_base == 0:

                if not self.initialized_output:
                    self.u_out = File('results/u_out.pvd')
                    self.p_out = File('results/p_out.pvd')
                    self.initialized_output = True

                if(not self.area_probe is None):
                    self.area_probe.dump(self.area_probe)
                self.u_out << self.flow.u_
                self.p_out << self.flow.p_


    def _str_(self):
        # printi("EnvBackward_step ---")
        print('')

    def close(self):
        self.ready_to_use = False

#-------------------------------------------------------------------------

    def reset(self):

        if self.inspection_params["show_all_at_reset"]:
            self.show_control()

        self.start_class()

        next_state = np.transpose(np.array(self.probes_values))
        # if self.verbose > 0:
        #     print(next_state)

        self.episode_number += 1

        return(next_state)



#------------------------------------------------------------------------

    def execute(self, actions=None):
        action = actions

        if self.verbose > 1:
            print("--- call execute ---")

        if action is None:
            if self.verbose > -1:
                print("carefull, no action given; by default, no jet!")
            
            nbr_contr = len(self.geometry_params["control_terms"])

            action = np.zeros((nbr_contr, ))

        if self.verbose > 2:
            print(action)
            
        self.previous_action = self.action
        self.action = action

        # to execute several numerical integration steps
        for crrt_action_nbr in range(self.number_steps_execution):

            # try to force a continuous / smoothe(r) control
            if "smooth_control" in self.optimization_params:
                # self.Qs += self.optimization_params["smooth_control"] * (np.array(action) - self.Qs)  
                # a linear change in the control
                if self.geometry_params['set_freq']:
                    [self.Qs,self.control_evolution] = np.array(self.previous_action) + (np.array(self.action) - np.array(self.previous_action)) / self.number_steps_execution * (crrt_action_nbr + 1)  
                    self.frequencies = pi/8*np.ones(1)  
                elif self.geometry_params['set_control']:
                    [self.Qs,self.frequencies] = np.array(self.previous_action) + (np.array(self.action) - np.array(self.previous_action)) / self.number_steps_execution * (crrt_action_nbr + 1)  
                    self.control_evolution = np.zeros(1)
                else:
                    [self.Qs, self.frequencies,self.control_evolution] = np.array(self.previous_action) + (np.array(self.action) - np.array(self.previous_action)) / self.number_steps_execution * (crrt_action_nbr + 1)  
                

            else:
                if self.geometry_params['set_freq']:
                    [self.Qs,self.control_evolution] = np.array(self.previous_action) + (np.array(self.action) - np.array(self.previous_action)) / self.number_steps_execution * (crrt_action_nbr + 1)  
                    self.frequencies = pi/8*np.ones(1)  
                elif self.geometry_params['set_control']:
                    [self.Qs,self.frequencies] = np.array(self.previous_action) + (np.array(self.action) - np.array(self.previous_action)) / self.number_steps_execution * (crrt_action_nbr + 1)  
                    self.control_evolution = np.zeros(1)
                else:
                    [self.Qs, self.frequencies,self.control_evolution] = np.array(self.previous_action) + (np.array(self.action) - np.array(self.previous_action)) / self.number_steps_execution * (crrt_action_nbr + 1)  
                
            # impose a zero net Qs
            if "zero_net_Qs" in self.optimization_params:
                if self.optimization_params["zero_net_Qs"]:
                    self.Qs = self.Qs - np.mean(self.Qs)
                    self.frequencies = self.frequencies - np.mean(self.frequencies)
                    self.control_evolution = self.control_evolution -np.mean(self.control_evolution)

            # evolve one numerical timestep forward
            
            if self.geometry_params['set_freq']:
                self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
            elif self.geometry_params['set_control']:
                self.u_, self.p_ = self.flow.evolve(np.concatenate((self.Qs, self.frequencies,self.control_evolution)))
            else:
                self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies,self.control_evolution])

            # displaying information that has to do with the solver itself
            self.visual_inspection()
            self.output_data()

            # we have done one solver step
            self.solver_step += 1

            # sample probes and drag
            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()

            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            # write to the history buffers
            self.write_history_parameters()


       
        next_state = np.transpose(np.array(self.probes_values))

        if self.verbose > 2:
            print(next_state)

        terminal = False

        if self.verbose > 2:
            print(terminal)

        reward = self.compute_reward()

        if self.verbose > 2:
            print(reward)

        if self.verbose > 1:
            print("--- done execute ---")

        return(next_state, terminal, reward)

        # return area

    def compute_reward(self):
        
        if(self.reward_function == 'recirculation_area'):
            return -self.area_probe.sample(self.u_, self.p_)
        elif(self.reward_function == 'max_recirculation_area'):
            return -self.area_probe.sample(self.u_, self.p_)
        else:
            raise RuntimeError("reward function {} not yet implemented".format(self.reward_function))

    def states(self):
        if self.output_params["probe_type"] == 'pressure':
            return dict(type='float',
                        shape=(len(self.output_params["locations"]) * self.optimization_params["num_steps_in_pressure_history"], )
                        )

        elif self.output_params["probe_type"] == 'velocity':
            return dict(type='float',
                        shape=(2 * len(self.output_params["locations"]) * self.optimization_params["num_steps_in_pressure_history"], )
                        )

    def actions(self):
        
        return dict(type='float',
                    shape=(len(self.geometry_params["control_terms"]), ),
                    min_value=self.optimization_params["min_value_jet_MFR"],
                    max_value=self.optimization_params["max_value_jet_MFR"])

    def max_episode_timesteps(self):
        return None