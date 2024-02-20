import os
import socket
import numpy as np
import csv

from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner

from simulation_base.set_env import resume_env, actuations_number

example_environment = resume_env(plot=False, dump=100, single_run=True)

deterministic = True

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

saver_restore = dict(directory=os.getcwd() + "/saver_data/")

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=example_environment, max_episode_timesteps=actuations_number,
    # TODO: actuations_number could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, estimate_terminal=True,  # ???
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    critic_network=network,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TensorFlow etc
    parallel_interactions=1,
    saver=saver_restore,
)

# restore_directory = './saver_data/'
# restore_file = 'model-40000'
# agent.restore(restore_directory, restore_file)
# agent.restore()
agent.initialize()



if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")

def one_run():
    print("start simulation")
    state = example_environment.reset()
    example_environment.render = True

    for k in range(10*actuations_number):
        #environment.print_state()
        action = agent.act(state, deterministic=deterministic, independent=True)
        state, terminal, reward = example_environment.execute(action)
    # just for test, too few timesteps
    # runner.run(episodes=10000, max_episode_timesteps=20, episode_finished=episode_finished)
    
    # Getting data from test_strategy.csv
    data = np.genfromtxt("saved_models/test_strategy.csv", delimiter=";")
    # This following line slices the loaded data array to exclude the first row and the first column 
    # We are taking away the simulation name and the indices, getting only the data.
    data = data[1:,1:] 
    #len(data)//2 calculates the index corresponding 
    #to the middle of the data array using integer division.(middle row)
    #and we are taking the second half of the rows
    m_data = np.average(data[len(data)//2:], axis=0)

    nb_jets = len(m_data)-3
    # Print statistics
    print("Single Run finished. AvgRecircArea : {}, AvgJetAmplitude : {}, AvgFrequency : {}".format(m_data[1], m_data[2],m_data[3]))

    name = "test_strategy_avg.csv"
    if(not os.path.exists("saved_models")):
        os.mkdir("saved_models")
    if(not os.path.exists("saved_models/"+name)):
        with open("saved_models/"+name, "w") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["Name", "RecircArea"] + ["Jet" + str(v) for v in range(nb_jets)])
            spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())
    else:
        with open("saved_models/"+name, "a") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())



if not deterministic:
    for _ in range(10):
        one_run()

else:
    one_run()
