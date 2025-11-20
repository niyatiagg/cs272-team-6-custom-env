
# CS272-Final Project - Team 6




## Custom Environment - CAR ACCIDENT (AccidentEnv)
## Authors

- [Niyati Aggarwal](https://www.github.com/niyatiagg)
- [Rajshri Ganesh Iyer](https://github.com/GRIYER26)
- [Lillian Zhang](https://github.com/lillianzhang-sjsu)


## Overview
Our custom environment [accident-v0] simulates a Car Accident scenario built on top of [Highway-Env](https://highway-env.farama.org/environments/highway/) and [Gymnasium](https://gymnasium.farama.org/). Here are the novel customizations we have introduced into our environment - 

    1. We have introduced a 2-car crash on the highway, halfway down the road, spread across 2 lanes. Although the location of the crash is fixed, the crash may occur in any 2 lanes chosen randomly. 
    
    2. We have also modified the reward functions to better suit a highway crash scenario.
## Objective
The main objective of the ego-vehicle is to react to the crash on the highway and respond appropriately in the following ways to ensure safe and efficient driving.

    1. Not colliding with other vehicles or the crashed  vehicles. 
    2. Moving away from the crash lanes. 
    2. Adjusting speed as it approaches crash, but ultimately driving at posted speed limits once away from crash zone.
    3. Not coming to a complete stop at any point.
    4. Not tailgating any vehicles at any point.
    4. Not driving off the road at any point.
## Reward Function
Our reward function retains some rewards from the original highway_env and introduces some additional rewards and penalties, adapted for a crash scenario. The rewards are designed to encourage safe and efficient driving.

    collision_reward :
    high_speed_reward :
    right_lane_reward :
    on_road_reward :
    reaction_reward : 
    tailgating_reward :
    job_well_done_reward :





## Termination & Truncation
The episode terminates if - 

    1. The ego vehicle drives off the road.
    2. The ego vehicle collides with another vehicle.

The episode truncates if - 

    1. The time limit is exceeded.
## Usage
### Install depedencies 
```pip install highway-env```

You can experience the environment in manual control mode by running 
```python run_custom_env.py```
## Citations
[1] E. Leurent, “An Environment for Autonomous Driving Decision-Making,” GitHub repository. [GitHub, 2018 Available Online](https://github.com/eleurent/highway-env)

[2] M. Towers et al., “Gymnasium.” [Zenodo, Mar. 2023 Available Online](https://zenodo.org/records/8127026)

