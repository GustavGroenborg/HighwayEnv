from __future__ import annotations
import math

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork, WeightedRoadnetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.utils import Vector
from highway_env.road.lanes.lane_utils import LaneType, LineType
from highway_env.road.lanes.abstract_lanes import AbstractLane

from highway_env.road.regulation import RegulatedRoad
from highway_env.network_builder import NetworkBuilder, StraightPath, CircularPath, Path
from highway_env.road.lanes.unweighted_lanes import StraightLane, SineLane, CircularLane


class RoadEnv(AbstractEnv):
    """
    A testing driving environment.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "LidarObservation",
                    "cells": 360,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [-10, 0, 5, 10, 20] #np.linspace(0, 30, 10)
                },
                "simulation_frequency": 15,
                "lanes_count": 2,
                "vehicles_count": 0,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 60,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "screen_height": 600,
                "screen_width": 1200,
            }
        )
        return config    

    def _reset(self) -> None:
            self._make_road()
            self._make_vehicles()
    
    def _make_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""

        net = WeightedRoadnetwork()
        nb = NetworkBuilder()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        left_turn = False
        right_turn = True
        radius = 20
        lane_width = 4
        
        """First highway circuit"""
        nb.add_multiple_nodes({
            "a:1" : [0, 4],
            "d:1" : [200, 4],
            "e:1" : [224, -20],
            "f:1" : [224, -220],
            "g:1" : [200, -244],
            "h:1" : [0, -244],
            "i:1" : [-24, -220],
            "j:1" : [-24, -20],
            
            "aa:1" : [0, 0],
            "dd:1" : [200, 0],
            "ee:1" : [220, -20],
            "ff:1" : [220, -220],
            "gg:1" : [200, -240],
            "hh:1" : [0, -240],
            "ii:1" : [-20, -220],
            "jj:1" : [-20, -20],
        })
        
        """Counter clockwise circuit (outer ring)"""
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("a:1", "d:1", (n,c), nb.get_weight(200, 80), LaneType.ROAD),
                StraightPath("e:1", "f:1", (n,c), nb.get_weight(200, 80), LaneType.ROAD),
                StraightPath("g:1", "h:1", (n,c), nb.get_weight(200, 80), LaneType.ROAD),
                StraightPath("i:1", "j:1", (n,c), nb.get_weight(200, 80), LaneType.ROAD),
            ],
            nb.PathType.CIRCULAR : [
                CircularPath("d:1", "e:1", 90, radius + 1 * lane_width, left_turn, (n,c), nb.get_weight(19, 80), LaneType.ROAD),  
                CircularPath("f:1", "g:1", 0, radius + 1 * lane_width, left_turn, (n,c), nb.get_weight(19, 80), LaneType.ROAD),  
                CircularPath("h:1", "i:1", -90, radius + 1 * lane_width, left_turn, (n,c), nb.get_weight(19, 80), LaneType.ROAD),  
                CircularPath("j:1", "a:1", 180, radius + 1 * lane_width, left_turn, (n,c), nb.get_weight(19, 80), LaneType.ROAD),  
            ]
        })
        
        nb.add_multiple_paths({
            nb.PathType.STRAIGHT : [
                StraightPath("dd:1", "aa:1", (s,c), nb.get_weight(200, 80), LaneType.ROAD),
                
                StraightPath("ff:1", "ee:1", (s,c), nb.get_weight(200, 80), LaneType.ROAD),
                
                StraightPath("hh:1", "gg:1", (s,c), nb.get_weight(200, 80), LaneType.ROAD),
                
                StraightPath("jj:1", "ii:1", (s,c), nb.get_weight(200, 80), LaneType.ROAD),
            ],
            nb.PathType.CIRCULAR : [
                CircularPath("ee:1", "dd:1", 180, radius + 0 * lane_width, right_turn, (s,c), nb.get_weight(19, 80), LaneType.ROAD),  
                CircularPath("gg:1", "ff:1", 90, radius + 0 * lane_width, right_turn, (s,c), nb.get_weight(19, 80), LaneType.ROAD),  
                CircularPath("ii:1", "hh:1", 0, radius + 0 * lane_width, right_turn, (s,c), nb.get_weight(19, 80), LaneType.ROAD),  
                CircularPath("aa:1", "jj:1", -90, radius + 0 * lane_width, right_turn, (s,c), nb.get_weight(19, 80), LaneType.ROAD),  
            ]
        })
        
        nb.build_paths(net)
        
        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        self.road = road
        return

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            ego_vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=0
                # spacing=self.config["ego_spacing"],
            )
            
            ego_vehicle = self.action_type.vehicle_class(
                self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
            )
            self.controlled_vehicles.append(ego_vehicle)
            self.road.vehicles.append(ego_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.check_collisions = False
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                
        for v in self.road.vehicles:  # Prevent early collisions
            if (
                v is not ego_vehicle
                and np.linalg.norm(v.position - ego_vehicle.position) < 20
            ):
                self.road.vehicles.remove(v)

        
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        MIN_REWARD = -1
        MAX_REWARD = 2.1
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    MIN_REWARD,
                    MAX_REWARD,
                ],
                [0, 1],
            )

        if self.config["normalize_reward"]:
            reward = np.clip(reward, 0, 1)
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
        }
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]