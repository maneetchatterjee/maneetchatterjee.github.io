"""
Multi-Robot Formation Control Environment with Dynamic Obstacles.

Robotics task for evaluating QEGAN and baseline models.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
import networkx as nx


class MultiRobotFormationEnv:
    """
    Environment for multi-robot formation control with obstacle avoidance.
    
    Task: Maintain desired formation while navigating through dynamic obstacles
    and minimizing energy consumption.
    """
    
    def __init__(
        self,
        num_robots: int = 10,
        workspace_size: Tuple[float, float] = (10.0, 10.0),
        num_obstacles: int = 5,
        formation_type: str = 'circle',
        dt: float = 0.1
    ):
        self.num_robots = num_robots
        self.workspace_size = workspace_size
        self.num_obstacles = num_obstacles
        self.formation_type = formation_type
        self.dt = dt
        
        # Robot state: [x, y, vx, vy]
        self.robot_states = None
        
        # Obstacles: [x, y, radius, vx, vy]
        self.obstacles = None
        
        # Desired formation
        self.target_formation = self._generate_formation()
        
        # Communication graph
        self.comm_graph = None
        
        self.reset()
    
    def _generate_formation(self) -> np.ndarray:
        """Generate desired formation positions."""
        positions = []
        
        if self.formation_type == 'circle':
            radius = 2.0
            for i in range(self.num_robots):
                angle = 2 * np.pi * i / self.num_robots
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append([x, y])
        
        elif self.formation_type == 'line':
            spacing = 0.5
            for i in range(self.num_robots):
                x = i * spacing - (self.num_robots - 1) * spacing / 2
                y = 0.0
                positions.append([x, y])
        
        elif self.formation_type == 'v_shape':
            spacing = 0.5
            for i in range(self.num_robots):
                if i < self.num_robots // 2:
                    x = -i * spacing
                    y = -i * spacing
                else:
                    j = i - self.num_robots // 2
                    x = j * spacing
                    y = -j * spacing
                positions.append([x, y])
        
        return np.array(positions)
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment to initial state."""
        # Initialize robots randomly
        self.robot_states = np.zeros((self.num_robots, 4))
        for i in range(self.num_robots):
            # Random position
            self.robot_states[i, 0] = np.random.uniform(0, self.workspace_size[0])
            self.robot_states[i, 1] = np.random.uniform(0, self.workspace_size[1])
            # Zero velocity initially
            self.robot_states[i, 2:4] = 0.0
        
        # Initialize dynamic obstacles
        self.obstacles = np.zeros((self.num_obstacles, 5))
        for i in range(self.num_obstacles):
            self.obstacles[i, 0] = np.random.uniform(1, self.workspace_size[0] - 1)
            self.obstacles[i, 1] = np.random.uniform(1, self.workspace_size[1] - 1)
            self.obstacles[i, 2] = np.random.uniform(0.3, 0.6)  # radius
            # Random velocity
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.1, 0.3)
            self.obstacles[i, 3] = speed * np.cos(angle)
            self.obstacles[i, 4] = speed * np.sin(angle)
        
        # Build communication graph
        self._update_communication_graph()
        
        return self._get_observation()
    
    def _update_communication_graph(self):
        """Update robot communication graph based on distance."""
        comm_range = 3.0  # Communication range
        
        # Build adjacency matrix
        adj_matrix = np.zeros((self.num_robots, self.num_robots))
        
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dist = np.linalg.norm(
                    self.robot_states[i, :2] - self.robot_states[j, :2]
                )
                if dist <= comm_range:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        # Convert to edge index format for PyTorch Geometric
        edge_list = []
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if adj_matrix[i, j] == 1:
                    edge_list.append([i, j])
        
        if len(edge_list) == 0:
            # If no connections, create a minimal connected graph
            for i in range(self.num_robots - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])
        
        self.comm_graph = np.array(edge_list).T
    
    def _get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current observation for the GNN.
        
        Returns:
            node_features: [num_robots, feature_dim]
            edge_index: [2, num_edges]
        """
        node_features = []
        
        for i in range(self.num_robots):
            # Robot's own state
            robot_state = self.robot_states[i].copy()
            
            # Relative position to formation center
            formation_center = np.mean(self.target_formation, axis=0)
            current_center = np.mean(self.robot_states[:, :2], axis=0)
            rel_to_center = current_center - formation_center
            
            # Distance to desired position in formation
            desired_pos = self.target_formation[i] + formation_center
            dist_to_desired = np.linalg.norm(robot_state[:2] - desired_pos)
            
            # Closest obstacle info
            min_obstacle_dist = float('inf')
            closest_obstacle_vel = np.zeros(2)
            for obs in self.obstacles:
                dist = np.linalg.norm(robot_state[:2] - obs[:2]) - obs[2]
                if dist < min_obstacle_dist:
                    min_obstacle_dist = dist
                    closest_obstacle_vel = obs[3:5]
            
            # Combine features
            features = np.concatenate([
                robot_state,  # [x, y, vx, vy] - 4 dims
                rel_to_center,  # [dx, dy] - 2 dims
                [dist_to_desired],  # 1 dim
                [min_obstacle_dist],  # 1 dim
                closest_obstacle_vel,  # [vx, vy] - 2 dims
            ])
            
            node_features.append(features)
        
        node_features = np.array(node_features)
        edge_index = self.comm_graph
        
        return node_features, edge_index
    
    def step(self, actions: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Control actions for each robot [num_robots, 2] (acceleration)
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        # Clip actions
        actions = np.clip(actions, -1.0, 1.0)
        
        # Update robot states
        for i in range(self.num_robots):
            # Update velocity
            self.robot_states[i, 2:4] += actions[i] * self.dt
            # Limit velocity
            vel_norm = np.linalg.norm(self.robot_states[i, 2:4])
            if vel_norm > 2.0:
                self.robot_states[i, 2:4] *= 2.0 / vel_norm
            
            # Update position
            self.robot_states[i, :2] += self.robot_states[i, 2:4] * self.dt
            
            # Keep within workspace
            self.robot_states[i, 0] = np.clip(
                self.robot_states[i, 0], 0, self.workspace_size[0]
            )
            self.robot_states[i, 1] = np.clip(
                self.robot_states[i, 1], 0, self.workspace_size[1]
            )
        
        # Update obstacles
        for i in range(self.num_obstacles):
            self.obstacles[i, :2] += self.obstacles[i, 3:5] * self.dt
            
            # Bounce off walls
            if self.obstacles[i, 0] <= 0 or self.obstacles[i, 0] >= self.workspace_size[0]:
                self.obstacles[i, 3] *= -1
            if self.obstacles[i, 1] <= 0 or self.obstacles[i, 1] >= self.workspace_size[1]:
                self.obstacles[i, 4] *= -1
        
        # Update communication graph
        self._update_communication_graph()
        
        # Calculate reward
        reward = self._calculate_reward(actions)
        
        # Check if done (collision or time limit)
        done, info = self._check_done()
        
        # Get next observation
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _calculate_reward(self, actions: np.ndarray) -> float:
        """Calculate reward for current state."""
        reward = 0.0
        
        # Formation maintenance reward
        formation_center = np.mean(self.target_formation, axis=0)
        current_center = np.mean(self.robot_states[:, :2], axis=0)
        
        formation_error = 0.0
        for i in range(self.num_robots):
            desired_pos = self.target_formation[i] + current_center
            actual_pos = self.robot_states[i, :2]
            error = np.linalg.norm(desired_pos - actual_pos)
            formation_error += error
        
        formation_error /= self.num_robots
        reward -= formation_error * 2.0
        
        # Obstacle avoidance penalty
        collision_penalty = 0.0
        for i in range(self.num_robots):
            for obs in self.obstacles:
                dist = np.linalg.norm(
                    self.robot_states[i, :2] - obs[:2]
                ) - obs[2]
                if dist < 0.2:  # Safety margin
                    collision_penalty += (0.2 - dist) * 10.0
        
        reward -= collision_penalty
        
        # Energy efficiency (penalize large actions)
        energy_cost = np.sum(np.square(actions)) / self.num_robots
        reward -= energy_cost * 0.1
        
        # Connectivity reward (maintain communication)
        num_edges = self.comm_graph.shape[1]
        min_edges = self.num_robots - 1
        if num_edges >= min_edges:
            reward += 1.0
        
        return reward
    
    def _check_done(self) -> Tuple[bool, dict]:
        """Check if episode is complete."""
        info = {
            'collision': False,
            'formation_achieved': False
        }
        
        # Check for collisions
        for i in range(self.num_robots):
            for obs in self.obstacles:
                dist = np.linalg.norm(
                    self.robot_states[i, :2] - obs[:2]
                ) - obs[2]
                if dist < 0.0:
                    info['collision'] = True
                    return True, info
        
        # Check if formation is achieved
        formation_center = np.mean(self.target_formation, axis=0)
        current_center = np.mean(self.robot_states[:, :2], axis=0)
        
        max_error = 0.0
        for i in range(self.num_robots):
            desired_pos = self.target_formation[i] + current_center
            actual_pos = self.robot_states[i, :2]
            error = np.linalg.norm(desired_pos - actual_pos)
            max_error = max(max_error, error)
        
        if max_error < 0.1:
            info['formation_achieved'] = True
        
        return False, info
    
    def render(self) -> str:
        """Simple text rendering of the environment."""
        state_str = f"Robots: {self.num_robots}\n"
        state_str += f"Formation Type: {self.formation_type}\n"
        
        formation_center = np.mean(self.target_formation, axis=0)
        current_center = np.mean(self.robot_states[:, :2], axis=0)
        
        formation_error = 0.0
        for i in range(self.num_robots):
            desired_pos = self.target_formation[i] + current_center
            actual_pos = self.robot_states[i, :2]
            error = np.linalg.norm(desired_pos - actual_pos)
            formation_error += error
        
        formation_error /= self.num_robots
        state_str += f"Avg Formation Error: {formation_error:.3f}\n"
        
        return state_str


def create_robot_env(num_robots: int = 10, formation_type: str = 'circle'):
    """Factory function to create robot environment."""
    return MultiRobotFormationEnv(
        num_robots=num_robots,
        workspace_size=(10.0, 10.0),
        num_obstacles=5,
        formation_type=formation_type,
        dt=0.1
    )
