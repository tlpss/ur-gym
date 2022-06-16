import stable_baselines3 as sb3
from stable_baselines3 import SAC, TD3, PPO
from ur_gym.pusher.state_pusher_sim import SimPushState, PushStateConfig, SimPushState2
import numpy as np

def evaluate(model,max_episode_len, num_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    episode_rewards = []
    episode_lengths = []
    for i in range(num_episodes):
        done = False
        obs = env.reset()
        episode_step = 0
        discounted_episode_reward = 0
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            print(f"action = {action}")
            obs, reward, done, info = env.step(action)
            print(f"{obs} - {reward} - {done}")
            env.render()
            discounted_episode_reward*= model.gamma
            discounted_episode_reward += reward
            episode_step+= 1
        episode_rewards.append(discounted_episode_reward)
        episode_lengths.append(episode_step)

    mean_episode_reward = np.mean(episode_rewards)
    mean_episode_lenght = np.mean(episode_lengths)
    succes = sum(np.array(episode_lengths) < max_episode_len) /num_episodes
    print("Mean reward:", mean_episode_reward, "avg episode duration:", mean_episode_lenght, "succes rate:", succes)

    return mean_episode_reward
if __name__ == "__main__":
    env = SimPushState()

    model = SAC("MlpPolicy", env, gamma= 0.2, verbose=2,learning_starts=100,learning_rate=2e-3, tensorboard_log="./sac_pusher_state_tb/", batch_size=256, device='cpu')

    evaluate(model,PushStateConfig.max_episode_steps,10,deterministic=False)

    model.learn(total_timesteps=5000, log_interval=50,eval_freq=-1)
    evaluate(model,PushStateConfig.max_episode_steps,10,deterministic=True)

"""
    for i in range(10):
        obs = env.reset()
        env.render()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            print(f"action = {action}")
            obs,reward,done,info = env.step(action)
            print(obs,reward,done)
            env.render()
"""

