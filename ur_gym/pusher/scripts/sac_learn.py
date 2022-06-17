import stable_baselines3 as sb3
from stable_baselines3 import SAC
from ur_gym.pusher.state_pusher import URPushState
import wandb
from wandb.integration.sb3 import WandbCallback
config = {
    "gamma" : 0.9,
    "lr": 2e-3,
    "learning_starts": 50,
    "batch_size": 128,
    "time_steps": 3000,
    "seed": 2022
}
env = URPushState()

#https://docs.wandb.ai/guides/integrations/other/stable-baselines-3
run = wandb.init(project = "ur_pusher", config=config, sync_tensorboard=True)

model = SAC("MlpPolicy", env, verbose=1, seed=config["seed"],learning_starts=config["learning_starts"],gamma=config["gamma"], learning_rate=config["lr"], tensorboard_log= "lr/ur_pusher_state_tb/", batch_size=config["batch_size"], device='cpu')

callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}")
model.learn(total_timesteps=config["time_steps"], log_interval=10)

model.save("sac_state_pusher")

run.finish()