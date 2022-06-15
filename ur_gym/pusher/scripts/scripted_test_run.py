"""simply python script that runs a few episodes on the environment using the scripted policy.
This can be used for testing or as a starting point for collecting demonstrations.
"""

from ur_gym.pusher.state_pusher import URPushState
import numpy as np


env = URPushState()
n_episodes = 10
for i in range(n_episodes):
    obs =  env.reset()
    done = False
    while not done:
        # get the scripted motion
        angle,length = env._calculate_optimal_primitive(env.goal_position)
        # normalize the inputs
        angle /= 2*np.pi
        length /= 0.2
        action = [angle, length]
        obs, reward, done, info = env.step(action)
        print(obs,reward,done)