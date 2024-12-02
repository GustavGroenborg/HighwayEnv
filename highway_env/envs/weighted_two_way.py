from __future__ import annotations

import numpy as np
from typing_extensions import override
from typing import Dict, Any

from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.road.road import Road, WeightedRoadnetwork
from highway_env.road.lanes import StraightLane, LineType, LaneType
from highway_env.road.lanes.lane_utils import vec

class WeightedTwoWayEnv(TwoWayEnv):
    @override
    def _create_road(self) -> None:
        """
        Create a weighted road network for the two-way environment.
        """
        net = WeightedRoadnetwork()
        
        # Road parameters
        length = 200  # Road length
        width = 4    # Lane width
        
        # Line types
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line_types = [[c, s], [n, c]]
        
        # Create right lane (primary driving lane)
        net.add_lane(
            "0", "1",
            StraightLane(
                [0, -width/2], 
                [length, -width/2], 
                line_types=line_types[0],
                width=width
            ),
            weight=1.0, # Primary lane with higher weight
            lane_type= Lane_Type.ROAD
        )
        
        # Create left lane (counter-flow or overtaking lane)
        net.add_lane(
            "0", "1", 
            StraightLane(
                [0, width/2], 
                [length, width/2], 
                line_types=line_types[1],
                width=width
            ),
            weight=0.8, # Secondary lane with lower weight
            lane_type= Lane_Type.ROAD
        )
        
        # Reverse direction lanes
        net.add_lane(
            "1", "0",
            StraightLane(
                [length, -width/2], 
                [0, -width/2], 
                line_types=line_types[0],
                width=width
            ),
            weight=1.0,
            lane_type= Lane_Type.ROAD
        )
        
        net.add_lane(
            "1", "0", 
            StraightLane(
                [length, width/2], 
                [0, width/2], 
                line_types=line_types[1],
                width=width
            ),
            weight=0.8,
            lane_type= Lane_Type.ROAD
        )
        
        # Create road with the network
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False)
        )

    def _reward(self, action: int) -> float:
        """
        Compute complex reward function.
        
        Args:
            action (int): Taken action
        
        Returns:
            float: Computed reward
        """
        reward = 0
        
        # Lane preference reward (based on lane weight)
        lane_index = self.vehicle.lane_index
        lane_weight = self.road.network.get_lane(lane_index).weight
        reward += lane_weight
        
        # High-speed lane preference
        forward_direction = self.vehicle.lane.direction
        speed_reward = np.clip(
            np.dot(self.vehicle.velocity, forward_direction) / 
            self.config.get("max_speed", 30), 
            0, 1
        )
        reward += speed_reward
        
        # Collision punishment
        if self.vehicle.crashed:
            reward -= 10
        
        return reward

    def _rewards(self, action: int) -> Dict[str, float]:
        """
        Compute detailed reward components.
        
        Args:
            action (int): Taken action
        
        Returns:
            Dict[str, float]: Reward components
        """
        lane_index = self.vehicle.lane_index
        lane_weight = self.road.network.get_lane(lane_index).weight
        
        rewards = {
            "lane_reward": lane_weight,
            "speed_reward": np.clip(
                np.dot(self.vehicle.velocity, self.vehicle.lane.direction) / 
                self.config.get("max_speed", 30), 
                0, 1
            ),
            "collision_penalty": -10 if self.vehicle.crashed else 0
        }
        
        return rewards

# Configuration example
config = {
    "duration": 60,
    "vehicles_count": 10,
    "max_speed": 30,
    "show_trajectories": False
}