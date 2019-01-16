import numpy as np
import math
from physics_sim import PhysicsSim

class my_task():
    """ Define a task enviorment, including goal and feedback provided to the  
    agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
        # State current position, velocity and angular velocity
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal: to land on the ground as dircetly below as possible to starting location in the sky ([0,0,10]. 
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 

    def get_reward(self):
        """Return reward based on current position."""
        # Penalize for euler angels, as the quadcopter should be as stable as possible.
        penalty = (1. - abs(math.sin(self.sim.pose[3])))
        penalty *= (1. - abs(math.sin(self.sim.pose[4])))
        penalty *= (1. - abs(math.sin(self.sim.pose[5])))
        
        # Penalize for distance from target position.
        # I.e., for not being directly above target (axes x&y), as well as for being high above it (axis z).
        delta = abs(self.sim.pose[:3] - self.target_pos)
        r = math.sqrt(np.dot(delta, delta)) 
        if(r > 0.01): decay = math.exp(-1/r) 
        else: decay = 0
        
        # Calculate reward after penalties.    
        reward = 1. - decay
        reward *= penalty
        
        done = False
        # If the quadcopter reached the ground...
        if self.sim.pose[2] <= self.target_pos[2]:
            done=True  
            reward += 1  # Reward successful landing.
            reward -= 0.5*(self.sim.v[2]**2)  # And penalize for landing speed.
                
        # Penalize crashes by providinig a negative reward.
        if done and self.sim.time < self.sim.runtime:
            reward = -1.
        
        return reward

    def getPose(self):
        """Returns current poition"""
        return self.sim.pose

    def step(self, rotor_speeds):
        """Obtains next state, reward & done values based on recent action."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) 
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        return next_state, reward, done

    def reset(self):
        """Resets the simulation to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
class Task():
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state