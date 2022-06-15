"""simply python script that runs a few episodes on the environment using random action policy.
This can (SHOULD!) be used for testing, to see if things don't break.
"""

import logging
from ur_gym.pusher.state_pusher import URPushState
import numpy as np 


logging.basicConfig(level=logging.DEBUG)
env = URPushState()


print(env._get_object_position())
n_episodes = 10
for i in range(n_episodes):
    obs =  env.reset()
    done = False
    while not done:
        angle = np.random.random()
        length = np.random.random() 
        action = [angle, length]
        obs, reward, done, info = env.step(action)
        print(obs,reward,done)
