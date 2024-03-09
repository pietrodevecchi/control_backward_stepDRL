results_1: 
First tryout with the most refined mesh, not completely executed due to too large computational time.

results_2: 
One of our first simulation we have launched, not significant improvements in diminuishing the recirculation area. In the trainig we performed 200 episodes, 60 actuations per episode, but the jet amplitude range was too low to have a real effect on the flow.

results_3:
Same training configuration as in results_2 but with an higher jet amplitude, slightly better results were achieved.

results_4:
This is the simulation were we achieved the best results. In the training phase we performed 180 episodes, 80 actuations per episode, using the smooth_control.

results_5:
Same training configuration as in results_3 making exception for a larger control_width that results in a better training, not as good as in results_4.

results_6:
Same training configuration as in results_4, but here the frequency was fixed at pi/8 and training was performed only on jet amplitude within a smaller range for convergence purposes.

results_7:
Similar setting configurations but with an increased number of episodes and actuations, also here similar results were obtained.
