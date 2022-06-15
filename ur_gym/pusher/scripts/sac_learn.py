import stable_baselines3 as sb3
from stable_baselines3 import SAC
from ur_gym.pusher.state_pusher import URPushState


env = URPushState()

# model = SAC("MlpPolicy", env, verbose=2,learning_starts=64,tensorboard_log="./sac_pusher_state_tb/", batch_size=64, device='cpu')
# model.learn(total_timesteps=1000, log_interval=4)
# model.save("sac_state_pusher")

model = SAC.load("sac_state_pusher",device="cpu")

print("done")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()