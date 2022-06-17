import imp
import pathlib
import stable_baselines3 as sb3
from stable_baselines3 import SAC
from ur_gym.pusher.state_pusher import URPushState
import wandb
import stable_baselines3.common.off_policy_algorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os 
from wandb.integration.sb3 import WandbCallback
import torch 

class MyCheckpointCallback(CheckpointCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0, use_wandb: bool = True):
        super().__init__(save_freq,save_path,name_prefix,verbose)
        self.use_wandb = use_wandb   

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.use_wandb:
                wandb.save(f"{path}.zip",base_path=self.save_path)
            if isinstance(self.model, stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm):
                buffer_filename = f"{path}_buffer"
                self.model.save_replay_buffer(buffer_filename)
                if self.use_wandb:
                    wandb.save(f"{buffer_filename}.pkl", base_path= self.save_path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True



if __name__ == "__main__":

    folder_path = pathlib.Path(__file__).parent
    config = {
        "gamma" : 0.9,
        "lr": 2e-3,
        "learning_starts": 100,
        "batch_size": 128,
        "time_steps": 3000,
        "seed": 2022
    }

    run_id = "14j6qrq0" # this indicates if the run should start a new run or not
    
    torch.manual_seed(config["seed"])

    env = URPushState()
    tb_path = folder_path / "ur_pusher_state_tb/"
    #https://docs.wandb.ai/guides/integrations/other/stable-baselines-3

    if run_id:
        checkpoint_path = folder_path / "checkpoints"/f"{run_id}/"

        run = wandb.init(project = "ur_pusher", config=config, sync_tensorboard=True, resume="must", id = run_id)
        steps = input("Give the checkpoint to load (identified by number of steps as indicated in the checkpoint file:")
        model = SAC.load(f"{checkpoint_path}/rl_model_{steps}_steps.zip", env = env,tensorboard_log= tb_path,device="cpu")
        model.env.reset() # because we specify "reset_num_timesteps = False", we need to reset the environment ourselves.
        model.load_replay_buffer(f"{checkpoint_path}/rl_model_{steps}_steps_buffer.pkl")

    else:
        run = wandb.init(project = "ur_pusher", config=config, sync_tensorboard=True)
        checkpoint_path = folder_path / "checkpoints"/f"{run.id}/"

        model = SAC("MlpPolicy", env, verbose=1, seed=config["seed"],learning_starts=config["learning_starts"],
        gamma=config["gamma"],learning_rate=config["lr"], tensorboard_log= tb_path, batch_size=config["batch_size"], device='cpu')

    wandb_callback = WandbCallback()
    checkpointcallback = MyCheckpointCallback(100, checkpoint_path)
    #eval_callback = EvalCallback(env, n_eval_episodes=2,eval_freq=2)
    

    # do not reset num timesteps when continuing learning, this will keep the logging consistent between runs.
    model.learn(total_timesteps=config["time_steps"], log_interval=2,callback=[checkpointcallback, wandb_callback],reset_num_timesteps = (run_id == None))

