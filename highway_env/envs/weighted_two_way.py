from __future__ import annotations

import numpy as np
from typing_extensions import override
from typing import Dict, Any

from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.road.road import Road, WeightedRoadnetwork
from highway_env.road.lanes.unweighted_lanes import StraightLane
from highway_env.road.lanes.lane_utils import LineType, LaneType

class WeightedTwoWayEnv(TwoWayEnv):
    @override
    def _make_road(self) -> None:
        """
        Create a weighted road network for the two-way environment.
        """
        net = WeightedRoadnetwork()
        
        # Road parameters
        length = 1000  # Road length
        width = 4    # Lane width
        
        # Line types
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line_types = [[c, s], [n, c]]
        
        # Create right lane (primary driving lane)
        net.add_lane(
            "a", "b",
            StraightLane(
                [0, -width/2], 
                [length, -width/2], 
                line_types=(n, n),
                width=width
            ),
            weight=1.0, # Primary lane with higher weight
            lane_type= LaneType.ROAD
        )
        
        # Create left lane (counter-flow or overtaking lane)
        net.add_lane(
            "a", "b", 
            StraightLane(
                [0, width/2], 
                [length, width/2], 
                line_types=(s, c),
                width=width
            ),
            weight=0.8, # Secondary lane with lower weight
            lane_type= LaneType.ROAD
        )
        
        # Reverse direction lanes
        net.add_lane(
            "b", "a",
            StraightLane(
                [length, -width/2], 
                [0, -width/2], 
                line_types=(n,c),
                width=width
            ),
            weight=1.0,
            lane_type= LaneType.ROAD
        )
        
        net.add_lane(
            "b", "a", 
            StraightLane(
                [length, width/2], 
                [0, width/2],
                line_types=(n,n),
                width=width
            ),
            weight=0.8,
            lane_type= LaneType.ROAD
        )
        
        # Create road with the network
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False)
        )

    @override
    def _rewards(self, action: int) -> Dict[str, float]:
        """
        Compute detailed reward components with emphasis on strategic overtaking.
        
        Args:
            action (int): Taken action
        
        Returns:
            Dict[str, float]: Reward components
        """
        _, _, lane_index = self.vehicle.lane_index
        
        # Determine lane characteristics
        is_left_lane = lane_index == 1  # Assuming lane 1 is the left lane
        is_right_lane = lane_index == 0  # Assuming lane 0 is the right lane
        
        # Base velocity reward
        speed_reward = np.clip(
            np.dot(self.vehicle.velocity, self.vehicle.lane.direction) /
            self.config.get("max_speed", 30),
            0, 1
        )
        
        # Overtaking reward
        overtaking_reward = 0
        if is_left_lane:
            # Reward for being in the left lane (overtaking lane)
            overtaking_reward = 0.5
            
            # Additional reward for speed difference if overtaking
            speed_diff_reward = max(0, speed_reward - 0.5)
            overtaking_reward += speed_diff_reward
        
        # Lane preference reward
        lane_preference_reward = 1 if is_right_lane else 0.5
        
        rewards = {
            "lane_reward": lane_preference_reward,
            "overtaking_reward": overtaking_reward,
            "speed_reward": speed_reward,
            "collision_penalty": -10 if self.vehicle.crashed else 0,
            "total_reward": (
                lane_preference_reward + 
                overtaking_reward + 
                speed_reward - 
                (10 if self.vehicle.crashed else 0)
            )
        }
        
        return rewards

# Configuration example
config = {
    "duration": 60,
    "vehicles_count": 20,
    "max_speed": 30,
    "show_trajectories": False
}