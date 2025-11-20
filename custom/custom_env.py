import numpy as np
import highway_env
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import LinearVehicle

# A simple class for crashed vehicles to place as hazards
class CrashedVehicle(Vehicle):
        
        def __init__(self, road: Road, position, heading=0):
            super().__init__(road, position, heading, speed=0, predition_type="zero_steering")
            self.crashed = True

# A modified version of HighwayEnv 
class AccidentEnv(AbstractEnv):

    # TODO: Change reward values in config
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "LidarObservation"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "speed_limit": 30,
                "vehicles_count": 10,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 20,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [10, 30],
                "normalize_reward": True,
                "offroad_terminal": False
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes, with a car accident halfway down its length."""
        road_network = RoadNetwork.straight_road_network(4, length=1000, speed_limit=self.config["speed_limit"])
        self.road = Road(network=road_network, record_history=self.config["show_trajectories"])

        # Randomly determine which lane to add crashed vehicles to
        self.crash_lane_index = self.road.np_random.choice([1, 2, 3])
        self.crash_lane = road_network.lanes_dict()[("0", "1", self.crash_lane_index)]

        crashed_vehicle_1 = CrashedVehicle(self.road, position=self.crash_lane.position(500, -2), heading=45)
        crashed_vehicle_2 = CrashedVehicle(self.road, position=self.crash_lane.position(505, 0), heading=-45)
        self.road.objects.append(crashed_vehicle_1)
        self.road.objects.append(crashed_vehicle_2)

    def _create_vehicles(self) -> None:
        """Add the agent-controlled vehicle and the other vehicles to the road."""
        self.controlled_vehicles = []
        agent = Vehicle.create_random(
            self.road,
            speed=25.0,
            lane_id=self.config["initial_lane_id"],
            spacing=self.config["ego_spacing"],
        )
        agent = self.action_type.vehicle_class(
            self.road, agent.position, agent.heading, agent.speed
        )
        self.controlled_vehicles.append(agent)
        self.road.vehicles.append(agent)
        self.agent_vehicle = agent

        for _ in range(self.config["vehicles_count"]):
            vehicle = LinearVehicle.create_random(
                self.road, spacing=1 / self.config["vehicles_density"]
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    # TODO: Change from HighwayEnv's version
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

        # Penalty for being in the same lane(s) as the crash, close to the crash
        reaction_reward = 0
        if self.agent_vehicle.lane_index[2] == self.crash_lane_index or self.agent_vehicle.lane_index[2] == self.crash_lane_index - 1:
            distance_from_crash = np.linalg.norm(self.agent_vehicle.position - self.road.objects[0].position)
            reaction_reward = min(0, (distance_from_crash - 40) / 80)

        # Penalty for tailgating
        forward_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.agent_vehicle, self.agent_vehicle.lane_index)
        if forward_vehicle is not None:
            distance_from_forward_vehicle = np.linalg.norm(self.agent_vehicle.position - forward_vehicle.position)
            tailgating_reward = min(0, (distance_from_forward_vehicle - 10) / 20)
        else:
            tailgating_reward = 0.0

        # Reward for job well done - if agent is in the right-most lane and successfully avoided the crash
        is_right = self.agent_vehicle.lane_index == 3
        clearance_bonus = 0.3 if is_right and self.agent_vehicle.position[0] > 510 else 0.0

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "reaction_reward": float(reaction_reward),
            "tailgating_reward": float(tailgating_reward),
            "job_well_done_reward" : float(clearance_bonus)
        }

    # TODO: Add a destination?
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

