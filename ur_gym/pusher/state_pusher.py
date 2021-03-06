from dataclasses import dataclass
from pathlib import Path
import pickle
from random import random
import time
import gym
from camera_toolkit.zed2i import Zed2i, sl
from camera_toolkit.aruco import get_aruco_marker_poses, get_aruco_marker_coords
from camera_toolkit.reproject_to_z_plane import reproject_to_ground_plane
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from typing import Any, Optional, Union, Tuple, List
import numpy as np
import cv2
from gym.spaces import Box
import logging


class PushStateConfig:
    """
    Config for all params that should be shared by a real robot and the simulated point mass version
    """

    # workspace is relative to the robot base.
    max_pushing_distance = 0.2
    robot_space_x_lower = -0.35
    robot_space_x_upper = 0.25
    robot_space_y_lower = -0.48
    robot_space_y_upper = -0.22

    goal_l2_margin = 0.02

    default_goal_in_robot_x = -0.050
    default_goal_in_robot_y = -0.350

    max_episode_steps = 20


class URPushState(gym.Env):
    """
    This is a very simple environment, mostly for testing on in-vivo learning with the UR Robot.

    The agent needs to push an item to the target location.
    The observations are the 2D position of the object to push and the 2D goal position.
    The actions are a direction and length for a "push primitive". (both continuous).

    For more information on how this gym was created, see the UR_state_pusher.md file.
    """

    # specify calibration aruco marker position in UR base frame
    aruco_in_robot_x = -0.0318
    aruco_in_robot_y = -0.3648
    robot_to_aruco_translation = np.array([aruco_in_robot_x, aruco_in_robot_y, 0.0])

    # properties  for UR Robot
    robot_eef_z = 0.02
    robot_eef_orientation_rotvec = [0, 3.14, 0.0]  # eef pointing down
    home_pose = [-0.13, -0.21, 0.30, 0, 3.14, 0]
    vel = 1.0
    acc = 0.8

    robot_flange_radius = 0.03
    robot_motion_margin = 0.02  # margin added to the start pose of the linear trajectory to avoid collisions with the object while moving down

    # Object Properties
    object_radius = 0.05
    object_height = 0.045

    def __init__(self, robot_ip: str = "10.42.0.161", random_goals=False) -> None:
        super().__init__()
        self.n_steps_in_episode = 0

        self.use_random_goals = random_goals
        # init robot interfaces
        self.robot_control = RTDEControl(robot_ip)
        self.robot_receive = RTDEReceive(robot_ip)

        # init desired Zed Camera
        self.camera = Zed2i(sl.RESOLUTION.HD2K, 15)

        # init goal position to default
        self.goal_position = np.array(
            [PushStateConfig.default_goal_in_robot_x, PushStateConfig.default_goal_in_robot_y]
        )

        # load camera to marker transform
        with open(Path(__file__).parent / "marker.pickle", "rb") as f:
            aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
        self.aruco_in_camera_transform = np.eye(4)
        self.aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
        self.aruco_in_camera_transform[:3, 3] = aruco_in_camera_position
        self.cam_matrix = self.camera.get_mono_camera_matrix()

        # define the Gym action spaces (loosely), required for SB3
        self.action_space = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-0.5, -0.5, -0.5, -0.5]), high=np.array([0.5, 0.5, 0.5, 0.5]), dtype=np.float32
        )

    def _get_observation(self) -> List:
        """
        gym observation
        """
        object_position = self._get_object_position()

        obs = object_position.tolist()
        obs.extend(self.goal_position)
        return obs

    def _get_robot_position(self):
        # TODO: move to UR interface
        return self.robot_receive.getActualTCPPose()

    def _get_object_position(self):
        """
        Get object pose by reading in image frame, detecting keypoint and reprojecting it on the Z = 0 plane.
        """
        img = self.camera.get_mono_rgb_image()
        img = self.camera.image_shape_torch_to_opencv(img)

        for i in range(3):
            image_coords = get_aruco_marker_coords(img, cv2.aruco.DICT_5X5_250)
            if image_coords is None:
                if i == 2:
                    raise ValueError("Could not detect the aruco marker after 3 attempts. Is the marker visible?")
                logging.error("could not detect object aruco marker")
                time.sleep(1.0)
            else:
                break
        aruco_frame_coords = reproject_to_ground_plane(image_coords, self.cam_matrix, self.aruco_in_camera_transform, height = URPushState.object_height)

        robot_frame_coords = aruco_frame_coords + URPushState.robot_to_aruco_translation

        if not self.position_is_in_object_space(robot_frame_coords[0][:2]):  # pushed object outside of goal_space
            raise ValueError(
                "Object is outside of workspace.. this should not happen and is probably caused"
                " by inaccurate object position which makes the primitive motion behave unexpected."
            )

        return robot_frame_coords[0][:2]

    def reset(self):
        self.n_steps_in_episode = 0
        # reset goal position
        if self.use_random_goals:
            raise NotImplementedError

        # reset object
        self._reset_object()

        observation = self._get_observation()

        return observation

    def _reset_object(self):
        """
        Find a random position in the object workspace and move the object to that position.
         Assumes perfect observations of the current position.
        """

        object_position = self._get_object_position()
        distance_to_target = np.linalg.norm(object_position - self.goal_position)

        # only reset if object is not at "meaningful start location" i.e. if the goal was reached
        if distance_to_target > PushStateConfig.goal_l2_margin:
            return

        new_object_pose = self.get_random_object_position(self.goal_position)
        logging.debug(f"resetting object to {new_object_pose}")

        if not self._move_object_to_position(new_object_pose):
            raise ValueError("reset failed due to invalid motion, which should not happen..")

    @staticmethod
    def get_random_object_position(goal_position: np.ndarray) -> np.ndarray:
        """
        Brute-force sample positions until one is in the allowed object workspace
        """
        while True:
            x = np.random.random() - 0.5
            y = np.random.random() * 0.25 - 0.45
            position = np.array([x, y])
            logging.debug(f"proposed object reset {position}")
            if URPushState._position_is_in_workspace(
                position,
                margin=1.1
                * (URPushState.object_radius + URPushState.robot_flange_radius + URPushState.robot_motion_margin),
            ) and not np.linalg.norm(position-goal_position) < PushStateConfig.goal_l2_margin:
                return position

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        """
        performs action,
        returns observation, reward, is episode finished?, info dict (empty)
        """
        self.n_steps_in_episode += 1

        normalized_angle, normalized_length = action
        angle = normalized_angle * 2 * np.pi
        length = normalized_length * PushStateConfig.max_pushing_distance
        logging.debug(f"taking action ({angle},{length})")
        valid_action = self._execute_primitive_motion(angle, length)

        new_observation = self._get_observation()
        new_object_position = self._get_object_position()

        distance_to_target = np.linalg.norm(new_object_position - self.goal_position)
        done = self._is_episode_finished(self.n_steps_in_episode,distance_to_target)

        # determine reward
        reward = self.calculate_reward(valid_action, distance_to_target)


        return new_observation, reward, done, {}

    def _execute_primitive_motion(self, angle: float, length: float) -> bool:
        """
        Do the "motion primitive": a push along the desired angle and over the specified distance

        To avoid collisions this is executed as:
        - move to pre-start pose
        - move to start pose
        - push
        - move to post-push pose
        - move to "out-of-sight" pose (home)


        angle: radians in [0,2Pi]
        lenght: value in [0, max_pushing_distance]

        Returns True after executing if the motion was allowed
        (start robot position is in the robot workspace, end object position is in the block workspace)
        and False otherwise.
        """

        # get current position of the object
        current_object_position = self._get_object_position()

        # determine primitive motion start and endpoint
        push_direction = np.array([np.cos(angle), np.sin(angle)])

        block_start_point = current_object_position
        robot_start_point = block_start_point - push_direction * (
            URPushState.object_radius + URPushState.robot_flange_radius + URPushState.robot_motion_margin
        )
        block_end_point = block_start_point + length * push_direction
        robot_end_point = block_end_point - push_direction * (
            URPushState.object_radius + URPushState.robot_flange_radius
        )

        logging.debug(f"motion primitive: (angle:{angle},len:{length} ) - {block_start_point} -> {block_end_point}")
        # calculate if the proposed primitive does not violate the robot's workspace

        if not self._position_is_in_workspace(robot_start_point):
            logging.debug(f"invalid robot startpoint for primitive {block_start_point} ->  {block_end_point}")
            return False
        if not self.position_is_in_object_space(block_end_point, margin=0.01):
            logging.debug(f"invalid  block endpoint for primitive {block_start_point} ->  {block_end_point}")
            return False

        # move to start pose
        self._move_robot(robot_start_point[0], robot_start_point[1], URPushState.robot_eef_z + 0.05)
        # execute
        self._move_robot(robot_start_point[0], robot_start_point[1], URPushState.robot_eef_z)
        self._move_robot(robot_end_point[0], robot_end_point[1], URPushState.robot_eef_z)

        # move back to home pose
        self._move_robot(robot_end_point[0], robot_end_point[1], URPushState.robot_eef_z + 0.05)
        self._move_robot(URPushState.home_pose[0], URPushState.home_pose[1], URPushState.home_pose[2])
        return True

    @staticmethod
    def _position_is_in_workspace(position: np.ndarray, margin: float = 0.0) -> bool:
        x, y = position

        if not (PushStateConfig.robot_space_x_lower + margin < x < PushStateConfig.robot_space_x_upper - margin):
            return False
        if not (PushStateConfig.robot_space_y_lower + margin < y < PushStateConfig.robot_space_y_upper - margin):
            return False
        return True

    @staticmethod
    def position_is_in_object_space(position: np.ndarray, margin: float = 0.0) -> bool:
        return URPushState._position_is_in_workspace(
            position,
            margin=URPushState.object_radius
            + URPushState.robot_flange_radius
            + URPushState.robot_motion_margin
            + margin,
        )

    def _move_robot(self, x, y, z):
        """Synchronous movement of the robot."""
        pose = [x, y, z]
        pose.extend(URPushState.robot_eef_orientation_rotvec)
        if not self.robot_control.isPoseWithinSafetyLimits(pose):
            raise ValueError("Pose outside of workspace")
        self.robot_control.moveL(pose, URPushState.vel, URPushState.acc)

        # make sure to wait till the robot is at its new location
        # even though documentation states this should be sync? #TODO: find out.
        while not self.robot_control.isSteady():
            time.sleep(0.01)
    @staticmethod
    def _is_episode_finished(n_steps_in_episode, distance_to_target: float):
        done = distance_to_target < PushStateConfig.goal_l2_margin  # goal reached
        done = done or n_steps_in_episode >= PushStateConfig.max_episode_steps
        return done

    def _move_object_to_position(self, position: np.ndarray) -> bool:
        """
        Move object to a desired position assuming perfect observation of the current position.
        """
        angle, length = self._calculate_optimal_primitive(position)
        return self._execute_primitive_motion(angle, length)

    def _calculate_optimal_primitive(self, position) -> Tuple[float, float]:
        """
        Calculate the optimal angle and length assuming perfect observation.
        """
        current_position = self._get_object_position()
        vector = position - current_position
        angle = np.arctan2(vector[1], vector[0])
        length = np.linalg.norm(vector)

        # from [-pi,pi ] to [0,2pi] for normalization
        if angle < 0:
            angle += 2 * np.pi
        return angle, length

    @staticmethod
    def calculate_reward(valid_action: bool, distance_to_target:float) -> float:
        # determine reward

        #TODO: change this to a potential function
        # also: it seemed to improve convergence if I added an additional bonus fo reaching the goal.
        max_distance = 0.5  # approx largest distance possible
        reward =  - distance_to_target / max_distance
        return reward


if __name__ == "__main__":
    env = URPushState()
    print(env._get_object_position())