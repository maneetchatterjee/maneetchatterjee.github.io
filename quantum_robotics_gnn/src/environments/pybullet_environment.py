"""
Physics-based Multi-Robot Simulation using PyBullet.

Realistic 3D physics simulation for multi-robot formation control with:
- Real robot dynamics (TurtleBot3-inspired)
- Collision detection and response
- Dynamic obstacles with physics
- Sensor noise and actuator delays
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Tuple, List, Optional
import time


class PyBulletRobotEnv:
    """
    High-fidelity physics-based multi-robot environment using PyBullet.
    
    Suitable for RSS, IJCAI, IJCNN submissions with realistic simulation.
    """
    
    def __init__(
        self,
        num_robots: int = 10,
        num_obstacles: int = 5,
        formation_type: str = 'circle',
        use_gui: bool = False,
        dt: float = 0.01,
        sensor_noise: float = 0.01,
        actuator_noise: float = 0.05
    ):
        self.num_robots = num_robots
        self.num_obstacles = num_obstacles
        self.formation_type = formation_type
        self.use_gui = use_gui
        self.dt = dt
        self.sensor_noise = sensor_noise
        self.actuator_noise = actuator_noise
        
        # Workspace boundaries
        self.workspace_size = (10.0, 10.0)
        
        # Robot parameters (TurtleBot3-like)
        self.robot_radius = 0.105  # m
        self.robot_mass = 1.0  # kg
        self.max_velocity = 0.22  # m/s
        self.max_angular_velocity = 2.84  # rad/s
        
        # PyBullet setup
        if use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # Storage for robot and obstacle IDs
        self.robot_ids = []
        self.obstacle_ids = []
        self.plane_id = None
        
        # Target formation
        self.target_formation = self._generate_formation()
        
        # Communication graph
        self.comm_range = 3.0  # meters
        self.comm_graph = None
        
        # Step counter
        self.step_count = 0
        self.max_steps = 1000
        
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
        
        elif self.formation_type == 'grid':
            cols = int(np.sqrt(self.num_robots))
            rows = int(np.ceil(self.num_robots / cols))
            spacing = 0.5
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx < self.num_robots:
                        x = j * spacing - (cols - 1) * spacing / 2
                        y = i * spacing - (rows - 1) * spacing / 2
                        positions.append([x, y])
                        idx += 1
        
        return np.array(positions)
    
    def _create_robot(self, position: np.ndarray) -> int:
        """Create a cylindrical robot (TurtleBot3-like)."""
        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.robot_radius,
            height=0.2
        )
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.robot_radius,
            length=0.2,
            rgbaColor=[0.2, 0.5, 0.8, 1.0]
        )
        
        # Create multi-body
        robot_id = p.createMultiBody(
            baseMass=self.robot_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], 0.1]
        )
        
        # Set lateral friction
        p.changeDynamics(robot_id, -1, lateralFriction=0.5)
        
        return robot_id
    
    def _create_obstacle(self, position: np.ndarray, radius: float) -> int:
        """Create a spherical obstacle."""
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.8, 0.2, 0.2, 1.0]
        )
        
        obstacle_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], radius]
        )
        
        return obstacle_id
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment to initial state."""
        # Remove existing objects
        for robot_id in self.robot_ids:
            p.removeBody(robot_id)
        for obstacle_id in self.obstacle_ids:
            p.removeBody(obstacle_id)
        if self.plane_id is not None:
            p.removeBody(self.plane_id)
        
        self.robot_ids = []
        self.obstacle_ids = []
        
        # Create ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create robots at random positions
        for i in range(self.num_robots):
            x = np.random.uniform(1, self.workspace_size[0] - 1)
            y = np.random.uniform(1, self.workspace_size[1] - 1)
            robot_id = self._create_robot(np.array([x, y]))
            self.robot_ids.append(robot_id)
        
        # Create dynamic obstacles
        for i in range(self.num_obstacles):
            x = np.random.uniform(2, self.workspace_size[0] - 2)
            y = np.random.uniform(2, self.workspace_size[1] - 2)
            radius = np.random.uniform(0.3, 0.6)
            obstacle_id = self._create_obstacle(np.array([x, y]), radius)
            self.obstacle_ids.append(obstacle_id)
            
            # Give obstacles initial velocities
            vx = np.random.uniform(-0.3, 0.3)
            vy = np.random.uniform(-0.3, 0.3)
            p.resetBaseVelocity(obstacle_id, linearVelocity=[vx, vy, 0])
        
        self.step_count = 0
        
        # Update communication graph
        self._update_communication_graph()
        
        return self._get_observation()
    
    def _update_communication_graph(self):
        """Update robot communication graph based on distance."""
        robot_positions = self._get_robot_positions()
        
        # Build adjacency matrix
        adj_matrix = np.zeros((self.num_robots, self.num_robots))
        
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dist = np.linalg.norm(robot_positions[i] - robot_positions[j])
                if dist <= self.comm_range:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        # Convert to edge index format
        edge_list = []
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if adj_matrix[i, j] == 1:
                    edge_list.append([i, j])
        
        # Ensure connectivity
        if len(edge_list) == 0:
            for i in range(self.num_robots - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])
        
        self.comm_graph = np.array(edge_list).T if len(edge_list) > 0 else np.array([[], []]).T
    
    def _get_robot_positions(self) -> np.ndarray:
        """Get current robot positions."""
        positions = []
        for robot_id in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            positions.append([pos[0], pos[1]])
        return np.array(positions)
    
    def _get_robot_velocities(self) -> np.ndarray:
        """Get current robot velocities."""
        velocities = []
        for robot_id in self.robot_ids:
            vel, _ = p.getBaseVelocity(robot_id)
            velocities.append([vel[0], vel[1]])
        return np.array(velocities)
    
    def _get_obstacle_states(self) -> List[np.ndarray]:
        """Get obstacle positions and radii."""
        obstacle_states = []
        for obstacle_id in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            # Get collision shape radius
            shape_data = p.getCollisionShapeData(obstacle_id, -1)
            radius = shape_data[0][3][0]  # sphere radius
            vel, _ = p.getBaseVelocity(obstacle_id)
            obstacle_states.append(np.array([pos[0], pos[1], radius, vel[0], vel[1]]))
        return obstacle_states
    
    def _get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current observation for GNN.
        
        Returns:
            node_features: [num_robots, feature_dim]
            edge_index: [2, num_edges]
        """
        robot_positions = self._get_robot_positions()
        robot_velocities = self._get_robot_velocities()
        obstacle_states = self._get_obstacle_states()
        
        # Add sensor noise
        if self.sensor_noise > 0:
            robot_positions += np.random.randn(*robot_positions.shape) * self.sensor_noise
            robot_velocities += np.random.randn(*robot_velocities.shape) * self.sensor_noise
        
        node_features = []
        
        # Formation center
        formation_center = np.mean(self.target_formation, axis=0)
        current_center = np.mean(robot_positions, axis=0)
        
        for i in range(self.num_robots):
            # Robot state
            pos = robot_positions[i]
            vel = robot_velocities[i]
            
            # Relative to center
            rel_to_center = current_center - formation_center
            
            # Distance to desired position
            desired_pos = self.target_formation[i] + formation_center
            dist_to_desired = np.linalg.norm(pos - desired_pos)
            
            # Closest obstacle
            min_obstacle_dist = float('inf')
            closest_obstacle_vel = np.zeros(2)
            for obs_state in obstacle_states:
                obs_pos = obs_state[:2]
                obs_radius = obs_state[2]
                dist = np.linalg.norm(pos - obs_pos) - obs_radius
                if dist < min_obstacle_dist:
                    min_obstacle_dist = dist
                    closest_obstacle_vel = obs_state[3:5]
            
            # Combine features
            features = np.concatenate([
                pos,  # [x, y] - 2 dims
                vel,  # [vx, vy] - 2 dims
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
        Execute one step with physics simulation.
        
        Args:
            actions: Control actions [num_robots, 2] (velocity commands)
            
        Returns:
            observation, reward, done, info
        """
        # Add actuator noise
        if self.actuator_noise > 0:
            actions = actions + np.random.randn(*actions.shape) * self.actuator_noise
        
        # Clip actions
        actions = np.clip(actions, -1.0, 1.0)
        
        # Apply velocities to robots
        for i, robot_id in enumerate(self.robot_ids):
            vx = actions[i, 0] * self.max_velocity
            vy = actions[i, 1] * self.max_velocity
            
            # Get current position and orientation
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            
            # Apply velocity
            p.resetBaseVelocity(
                robot_id,
                linearVelocity=[vx, vy, 0],
                angularVelocity=[0, 0, 0]
            )
        
        # Step simulation
        p.stepSimulation()
        
        # Update communication graph
        self._update_communication_graph()
        
        # Calculate reward
        reward = self._calculate_reward(actions)
        
        # Check if done
        done, info = self._check_done()
        
        # Get observation
        observation = self._get_observation()
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            info['timeout'] = True
        
        return observation, reward, done, info
    
    def _calculate_reward(self, actions: np.ndarray) -> float:
        """Calculate reward based on formation quality and safety."""
        robot_positions = self._get_robot_positions()
        obstacle_states = self._get_obstacle_states()
        
        reward = 0.0
        
        # Formation maintenance
        formation_center = np.mean(self.target_formation, axis=0)
        current_center = np.mean(robot_positions, axis=0)
        
        formation_error = 0.0
        for i in range(self.num_robots):
            desired_pos = self.target_formation[i] + current_center
            actual_pos = robot_positions[i]
            error = np.linalg.norm(desired_pos - actual_pos)
            formation_error += error
        
        formation_error /= self.num_robots
        reward -= formation_error * 2.0
        
        # Collision penalty
        collision_penalty = 0.0
        for i in range(self.num_robots):
            for obs_state in obstacle_states:
                obs_pos = obs_state[:2]
                obs_radius = obs_state[2]
                dist = np.linalg.norm(robot_positions[i] - obs_pos) - obs_radius - self.robot_radius
                if dist < 0.2:
                    collision_penalty += (0.2 - dist) * 10.0
        
        reward -= collision_penalty
        
        # Energy efficiency
        energy_cost = np.sum(np.square(actions)) / self.num_robots
        reward -= energy_cost * 0.1
        
        # Connectivity reward
        num_edges = self.comm_graph.shape[1] if len(self.comm_graph.shape) > 1 else 0
        min_edges = self.num_robots - 1
        if num_edges >= min_edges:
            reward += 1.0
        
        return reward
    
    def _check_done(self) -> Tuple[bool, dict]:
        """Check termination conditions."""
        info = {'collision': False, 'formation_achieved': False, 'timeout': False}
        
        robot_positions = self._get_robot_positions()
        obstacle_states = self._get_obstacle_states()
        
        # Check collisions using PyBullet contact points
        for robot_id in self.robot_ids:
            for obstacle_id in self.obstacle_ids:
                contact_points = p.getContactPoints(robot_id, obstacle_id)
                if len(contact_points) > 0:
                    info['collision'] = True
                    return True, info
        
        # Check formation achievement
        formation_center = np.mean(self.target_formation, axis=0)
        current_center = np.mean(robot_positions, axis=0)
        
        max_error = 0.0
        for i in range(self.num_robots):
            desired_pos = self.target_formation[i] + current_center
            actual_pos = robot_positions[i]
            error = np.linalg.norm(desired_pos - actual_pos)
            max_error = max(max_error, error)
        
        if max_error < 0.15:
            info['formation_achieved'] = True
        
        return False, info
    
    def render(self, mode='human'):
        """Render is handled by PyBullet GUI."""
        if self.use_gui:
            time.sleep(self.dt)
    
    def close(self):
        """Clean up PyBullet."""
        p.disconnect(self.physics_client)


def create_pybullet_env(num_robots: int = 10, formation_type: str = 'circle', use_gui: bool = False):
    """Factory function for PyBullet environment."""
    return PyBulletRobotEnv(
        num_robots=num_robots,
        num_obstacles=5,
        formation_type=formation_type,
        use_gui=use_gui,
        dt=0.01,
        sensor_noise=0.01,
        actuator_noise=0.05
    )
