File Explanations :

jet_bcs.py: 

Contains the implementation of an object (JetBCValue) that computes the field at the Dirichlet boundary used as control jet. The two parameters that will be controlled by the agent are the amplitude and the frequency of the fluid jet impulse, that results in a sinusoidal function of time, directed at 45 degrees with respect to the horizontal x-direction.
The object is then used in flow_solver.py to impose boundary conditions when the spaces of the solution are defined.

flow_solver.py: 

Defines the object FlowSolver where all the CFD simulation is set. Dolfin is at the base of the code and at the initialization of the solver we build the spaces on the mesh for pressure and velocity with their boundary conditions and the solvers through algebraic Chorin Temam method.
There is the possibility of using a direct approach (LU factorization) or an iterative method based on the conjugate gradient.
Furthermore the boundary condition on the segment dedicated to the control is linked to a JetBCValue object (defined in jet_bcs.py). 
The main function of the object is evolve() that computes the solution at the next timestep updating before the jet BCs through the argument jet_bc_values passed to the function by the agent.


EnvBackward_step.py:

In this script the principal object of the library is defined, a class that inherits from the Tensorforce Environment object and overrides the principal feature such as state actions and reward implementations in the context of backward step CFD episodes. In addition, a wide range of visualization and saving output functions are introduced, to store data and plots for postprocessing and results analysis. 
Class is initialized basically in two ways, the first called in simulation folder with the remesh flag set to True, to simulate the baseline and set the starting point for each episode that will be used for control and learning. The second possibility, the one that is invoked during learning phase, calls the class initialization through the reading of a series of files that contains mesh data, initial conditions for the solver (u_init and p_init) and a dictionary with the parameters (dict_history_parameters.pkl).
The state of the learning consists in the values of our solution (it can be chosen to consider pressure or velocity) onto the probes, the action is a 2-dimensional value that contributes to the amplitude and the frequency of the jet on the boundary condition and the reward function is linked to the portion of the domain where velocity is under a threshold (in our case if negative) that is clearly proportional to the extent of the recirculation area. The implementation of this last one can be found in probes.py script.


test_cfd_simulation.py;

Here a parameter tuning on the solvers is performed, 
These are the solvers available for Krylov methods:
bicgstab      		|  Biconjugate gradient stabilized method 
cg            	 	|  Conjugate gradient method 
default        		|  default Krylov method
gmres         	 	|  Generalized minimal residual method
minres         		|  Minimal residual method
richardson     		|  Richardson method
tfqmr          		|  Transpose-free quasi-minimal residual method

These are the available preconditioners:
amg              		|  Algebraic multigrid
default          		|  default preconditioner
hypre_amg       		|  Hypre algebraic multigrid (BoomerAMG)
hypre_euclid     		|  Hypre parallel incomplete LU factorization
hypre_parasails  	        |  Hypre parallel sparse approximate inverse 
icc              		|  Incomplete Cholesky factorization
ilu              		|  Incomplete LU factorization
jacobi          	 	|  Jacobi iteration 
none             		|  No preconditioner
petsc_amg        		|  PETSc algebraic multigrid
sor              		|  Successive over-relaxation

generate_msh.py :

The file contains the function generate_mesh, which is called in 'our_EnvBackward_step.py' when the remesh attribute is turned on.
In the body of the function, through the flag variable "change_parameters", it is possible to update or change some of the geometry parameters without modifying the .geo file . For us it was useful, in order to understand the impact of the jet, to have the possibility of updating the variable "control_width" which of course need the modification also for the attribute "length_before_control".
The function then call through the attribute subprocess.call() gmsh to generate the mesh.


msh_convert.py : 

The file is basically the same as the one realized by Rabault in the original code provided for the Cylinder problem : the key is to obtain at the end of the process a .h5 file starting from the .msh file. This is obtained by meaning of calling dolfin-convert (we tried to modify it by using meshio-convert, but it caused problems) for a first conversion from .msh to .xml and it is completed by using the function HDF5File().

probes.py :

In this script there is the initialization of the probes for booth the velocity and the pressure calculus in the positions chosen in env.py. The two classes VelocityProbeValues and PressureProbeValues are child_classes of Probe_Evaluator from which they inherits the "sample" method through which the values for each timestep are calculated. Probe_Evaluator builds objects which, once localized the cells containing the probes, evaluate the basis function for the cell dofs, which are then saved and updated with the velocity or pressure value once the sample method is called. 
Moreover in this script it is also developed the TotalRecirculationArea class, whose objects provide the calculus of the global area of cells having (in their dofs) at least one values of the computed horizontal velocity below a given threshold (for us 0, so horizontal velocity opposite w.r.t the flow direction). This function is the one used as reward function for the learning. 


env.py :

In this script it's initialized the example environment called in the script lauch_parallel_training.py through the function resume_env.
The fundamental parameters for the simulations are all initialized here: both the geometrical ones but also the ones necessary for the training as the maximum and the minimum value for forcing jet and for the frequency, the timestep for each solver object iteration and the choice of reward function.

Freefem_CFD_code.edp :

This code is not part of the library we developed but it is the first approach we had in trying to compute a CFD simulation of the backward_step control problem. It is highly commented as it could serve as a first reading to understand the problem and the aim of the code we developed afterwards.
In particular the classes flow_solver.py and jet.bcs are built following the principles first used for this simulation. 
We took advantage of the FreeFem solver as it interface is simple and it is a tool we already knew from all of us as it was used in our previous CFD courses. 
The .edp format is the one recognized by FreeFem, and so the file it's ready to be directly run.
