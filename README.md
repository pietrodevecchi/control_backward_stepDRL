# Files and Directories explanation :

## control_backward_stepDRL:

This is the core of the project, it contains all the .py files that are called when running training_launcher.sh and here all the results will be stored. Another README file is provided to better understand the role of each python script.

mesh_ready_to_use: 

This directory contains the meshes we used to run our simulations. We provide this files to make easier the approach to our simulations since the gmsh's libraries can be quite annoying to confront with.

results_past_simulations:

Here we provide some of the outputs from our simulations, like velocity, recirculation area and pessure in a .vtu format. One can easily visualize them through Paraview.

Freefem_CFD_code.edp :

This code is not part of the library we developed but it is the first approach we had in trying to compute a CFD simulation of the backward_step control problem. It is highly commented as it could serve as a first reading to understand the problem and the aim of the code we developed afterwards. In particular the classes flow_solver.py and jet.bcs are built following the principles first used for this simulation. We took advantage of the FreeFem solver as it interface is simple and it is a tool we already knew from all of us as it was used in our previous CFD courses. The .edp format is the one recognized by FreeFem, and so the file it's ready to be directly run.
