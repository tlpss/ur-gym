import logging
import time
from typing import Tuple, List

from ur_gym.pusher.state_pusher import PushStateConfig, URPushState
import gym
from gym.spaces import Box
import numpy as np
import pygame

class SimPushState(gym.Env):
    pygame_scale = 2000
    def __init__(self) -> None:
        self.object_position = np.array([0.0, 0.0])
        self.goal_position = np.array([PushStateConfig.default_goal_in_robot_x, PushStateConfig.default_goal_in_robot_y])
        self.n_steps_in_episode = 0
        self.done = False

        # define the Gym action spaces (loosely), required for SB3
        self.action_space = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-0.5, -0.5, -0.5, -0.5]), high=np.array([0.5, 0.5, 0.5, 0.5]), dtype=np.float32
        )

        pygame.init()
        self.screen = None

    def _get_object_position(self) -> np.ndarray:
        return self.object_position

    def _get_observation(self) -> np.ndarray:
        observation = self.object_position.tolist()
        observation.extend(self.goal_position)
        return observation

    def reset(self):
        self.object_position = URPushState.get_random_object_position(self.goal_position)
        self.n_steps_in_episode = 0
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.n_steps_in_episode += 1
        normalized_angle, normalized_length = action
        angle = normalized_angle * 2 * np.pi
        length = normalized_length * PushStateConfig.max_pushing_distance
        valid_action = self._execute_primitive_motion(angle,length)

        new_observation = self._get_observation()
        new_object_position = self._get_object_position()

        distance_to_target = np.linalg.norm(new_object_position - self.goal_position)
        done = URPushState._is_episode_finished(self.n_steps_in_episode, distance_to_target)
        self.done = done
        reward = URPushState.calculate_reward(valid_action, distance_to_target)


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
        if mode =="human":
            if self.screen is None:
                self.screen = pygame.display.set_mode([(PushStateConfig.robot_space_x_upper-PushStateConfig.robot_space_x_lower)*self.pygame_scale, (PushStateConfig.robot_space_y_upper-PushStateConfig.robot_space_y_lower)*self.pygame_scale])
            self.screen.fill((255,255,255))
            pygame.draw.circle(self.screen, (0,255,0),tuple(self.transform_coords_to_pygame_frame(self.goal_position)), PushStateConfig.goal_l2_margin*self.pygame_scale)
            pygame.draw.circle(self.screen, (255*self.done,0,(1-self.done)*255),tuple(self.transform_coords_to_pygame_frame(self.object_position)),URPushState.object_radius * self.pygame_scale)
            pygame.display.flip()
            time.sleep(0.1)
        else:
            print(f"{self.object_position} -> {self.goal_position}")

    @staticmethod
    def transform_coords_to_pygame_frame(coords: np.ndarray) -> np.ndarray:
        """
        pygame is top-left with y-axis point down. Robot frame is y pointing up and "top-center"
        """

        pygame_coords = np.copy(coords)
        pygame_coords[0] -= PushStateConfig.robot_space_x_lower
        pygame_coords[1] -= PushStateConfig.robot_space_y_upper
        pygame_coords[1] *= -1

        pygame_coords *= SimPushState.pygame_scale
        return pygame_coords


class SimPushState2(SimPushState):
    """
    action space is now the displacement vector.
    """
    
    def __init__(self):
        super(SimPushState2, self).__init__()
        self.action_space = Box(low=np.array([-0.1,-0.03]), high= np.array([0.1,0.03]))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.n_steps_in_episode += 1
        action = np.copy(action)
        valid_action = self._execute_primitive_motion(action)

        new_observation = self._get_observation()
        new_object_position = self._get_object_position()

        distance_to_target = np.linalg.norm(new_object_position - self.goal_position)
        done = URPushState._is_episode_finished(self.n_steps_in_episode, distance_to_target)
        self.done = done
        reward = URPushState.calculate_reward(valid_action, distance_to_target)

        return new_observation, reward, done, {}

    def _execute_primitive_motion(self, translation) -> bool:
        block_end_point = self._get_object_position() + translation
        if not URPushState.position_is_in_object_space(block_end_point):
            return False

        self.object_position = block_end_point
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    env = SimPushState2()
    for i in range(10):
        env.reset()
        env.render()
        done = False
        while not done:
            action = env.action_space.sample()
            print(action)
            obs,reward, done, info = env.step(action)
            env.render()
            time.sleep(0.1)
            print(obs,reward,done)

