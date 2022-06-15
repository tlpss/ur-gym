from typing import Tuple, List

from ur_gym.pusher.state_pusher import PushStateConfig, URPushState
import gym
from gym.spaces import Box
import numpy as np


class SimPushState(gym.Env):
    def __init__(self) -> None:
        self.object_position = np.array([0.0, 0.0])
        self.goal_position = [PushStateConfig.default_goal_in_robot_x, PushStateConfig.default_goal_in_robot_y]
        self.n_steps_in_episode = 0

        # define the Gym action spaces (loosely), required for SB3
        self.action_space = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-0.5, -0.5, -0.5, -0.5]), high=np.array([0.5, 0.5, 0.5, 0.5]), dtype=np.float32
        )

    def _get_object_position(self) -> np.ndarray:
        return self.object_position

    def _get_observation(self) -> np.ndarray:
        observation = self.object_position.tolist()
        observation.extend(self.goal_position)
        return observation

    def reset(self):
        self.object_position = URPushState.get_random_object_position()
        self.n_steps_in_episode = 0
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        normalized_angle, normalized_length = action
        angle = normalized_angle * 2 * np.pi
        length = normalized_length * PushStateConfig.max_pushing_distance
        valid_action = self._execute_primitive_motion(angle,length)

        new_observation = self._get_observation()
        new_object_position = self._get_object_position()

        distance_to_target = np.linalg.norm(new_object_position - self.goal_position)
        done = URPushState._is_episode_finished(self.n_steps_in_episode, distance_to_target)

        reward = URPushState.calculate_reward(valid_action, distance_to_target)

        self.n_steps_in_episode += 1

        return new_observation, reward, done, {}

    def _is_episode_finished(self, distance_to_target: float, object_position: np.ndarray):
        done = distance_to_target < PushStateConfig.goal_l2_margin  # goal reached
        done = done or self.n_steps_in_episode >= PushStateConfig.max_episode_steps
        return done

    def _execute_primitive_motion(self, angle: float, length: float) -> bool:
        current_object_position = self._get_object_position()

        # determine primitive motion start and endpoint
        push_direction = np.array([np.cos(angle), np.sin(angle)])
        block_end_point = current_object_position + length * push_direction
        if not URPushState.position_is_in_object_space(block_end_point):
            return False

        self.object_position = block_end_point
        return True

    def render(self, mode="human"):
        print(f"{self.object_position} -> {self.goal_position}")
if __name__ == "__main__":
    env = SimPushState()
    for i in range(1100):
        env.reset()
        done = False
        while not done:
            action = np.random.random(2)
            print(action)
            obs,reward, done, info = env.step(action)
            print(obs,reward,done)

