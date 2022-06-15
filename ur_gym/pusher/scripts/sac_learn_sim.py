import stable_baselines3 as sb3
from stable_baselines3 import SAC
from ur_gym.pusher.state_pusher_sim import SimPushState

if __name__ == "__main__":
    env = SimPushState()

    model = SAC("MlpPolicy", "Pendulum-v1", verbose=2,learning_starts=100, tensorboard_log="./sac_pusher_state_tb/", batch_size=256, device='cuda')
    model.learn(total_timesteps=100000, log_interval=4,eval_freq=10)


    print("done")
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #       obs = env.reset()