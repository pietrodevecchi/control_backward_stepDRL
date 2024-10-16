# Files and Directories explanation :

## control_BackwardStep:

This is the core of the project, it contains all the .py files that are called when running training_launcher.sh and here all the results will be stored. Another README file is provided to better understand the role of each python script.

## extra_material:

Here some insights are provided about meshes, past results and other utilities.

# Download 

We suggest to download the docker image components from the directory `Docker`, rigorously in the same destination folder.

At this point the image can be reconstructed through the command:

  ```bash
  pc@user cat bstep_control_image.tar_part.?? > bstep_control_image.tar
  ```

The image can be loaded into the Docker environment using:

  ```bash
  pc@user docker load < bstep_control_image.tar

  ```

Make sure the image you intend to use is available on your system:

  ```bash
  pc@user docker images
  ```

Now create and start a new container from your image:

  ```bash
  pc@user docker run -ti --name NAME_OF_THE_CONTAINER bstep_control_image:latest /bin/bash -l
  ```

The library is now ready to use, with the main directory at `/home/fenics/local/Bstep_DRLcontrol`.

# Instructions to run:

## 1. Initiating the Learning Process

- Start the learning process from the main `control_BackwardStep/` folder using the command

  ```bash
  pc@user control_BackwardStep $ ./script_launch_parallel.sh session_name first_port num_servers
  ```

-Upon starting, the tmux session divides the terminal to track the learning process across all servers. Each parallel server correlates with an environment, creating and utilizing `env_NUMBER_OF_SERVER/` folders to store outputs and results.

-The latest episode and the final reward from each episode can be viewed within in `results/` and `saved_models/` folders.

-Additionally, a checkpoint is generated in `saver_data/`, which is useful for pausing and resuming the learning process. Note that this checkpoint must be deleted if you wish to conduct a completely new optimization.

## 2. Setting Up the Geometry

- Begin by defining the geometry of the problem you aim to control within the `.geo` file. Specifically, ensure to hard-code the labels for the boundary conditions required for the CFD simulation. 
- By default, the geometry for a backward step channel, used in our test code, is located in `backward_facing_step.template_geo`. Additionally, the `mesh_ready_to_use` folder contains other meshes with various boundary conditions and refinements for more customized setups.

## 3. Configuring Parameters

- Modify parameters related to both the learning process (such as the number of actuations, the amplitude range for actions, duration of episodes, etc.) and the CFD simulation (solver type, preconditioners, Reynolds number, probe positions, etc.) through the `simulation_base/set_env.py` script.
- The default parameters are set to the values optimized through our tuning process that can be found also in the report.

## 4. Generating Mesh and Running Baseline Simulation

- Within the `simulation_base/` folder, execute the `launch_mesh_and_baseline.py` file from the terminal:
  
  ```bash
  pc@user simulation_base $ python3 launch_mesh_and_baseline.py
  ```

This command generates and converts the mesh from the .geo file and performs a lengthy simulation without any actions to achieve a stable state, serving as the starting point for each learning episode. It could be necessary to set the path with gmsh that sometimes creates problems.




