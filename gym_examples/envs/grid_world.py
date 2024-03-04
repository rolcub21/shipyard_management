import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.locals import *
from gymnasium.spaces import Dict, Box, MultiBinary
import scipy
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
import heapq
import math



class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", num_targets=4, size=6):
        self.size_rows = size  # The size of the square grid
        self.size_cols = size 
        #self.window_size = 512  # The size of the PyGame window
        self.window_width = (self.size_cols*100)
        self.window_height = self.size_rows*100
        self.screen_width = size*100
        self.screen_height = size*100
        self.number_cells = size
        self.distance_to_target = 0.1
        self.direction =  np.array([0, 0])
        self.num_targets = num_targets
        self.distances_door_to_destination= 0
        self.distances_targets_to_door = 0
        self.is_carrying_target = False 
        self._last_inside_reward = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size_rows -1 , shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size_rows -1 ,shape=(self.num_targets, 2), dtype=int),
                "target_picked_up": MultiBinary(self.num_targets),
                "target_dropped_off": MultiBinary(self.num_targets),
                "destination": spaces.Box(0, self.size_rows - 1, shape=(self.num_targets,2), dtype=int),
            }
        )


        # We have 6 actions, corresponding to "right", "up", "left", "down", "right", 
        #two actions for picking up and dropping off the target
        self.action_space = spaces.Discrete(6)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.screen = None
        self.clock = None

    def get_direction(self, action):
        action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]), # left
            3: np.array([0, -1]), # up
        }
        return action_to_direction.get(action)

    def _get_obs(self):
        
        # Ensure agent location is a numpy array (assuming it's a Box space)
        agent_location = np.array(self._agent_location, dtype=int)

        # Ensure target locations are a numpy array (assuming it's a Box space)
        # This conversion assumes self._target_location is list-like and already structured correctly;
        # adjust as necessary to match the actual structure of self._target_location
        target_location = np.array(self._target_location, dtype=int)


        # Ensure target_picked_up is a numpy array (assuming it's a MultiBinary space)
        # This line is correct; just make sure self._target_picked_up is structured as a flat list
        # of binary values (0 or 1) that matches the number of targets
        target_picked_up = np.array(self._target_picked_up, dtype=np.int8)


        target_dropped_off = np.array(self._target_dropped_off, dtype=np.int8)

        # Ensure destination is a numpy array (assuming it's a Box space)
        destination = np.array(self._destination, dtype=int)

        return {
            "agent": agent_location,
            "target": target_location,
            "target_picked_up": target_picked_up,
            "target_dropped_off": target_dropped_off,
            "destination": destination
        }

    def _get_info(self): 
        # Calculate distances from each target to the destination
        distances_targets_to_door= np.mean([np.linalg.norm(np.array(target_loc) - np.array(destination), ord=1)
            for target_loc, destination in zip(self._target_location, self._destination)
        ])

        distances_door_to_destination = np.mean([np.linalg.norm(np.array(destination) - np.array([self.size_rows - 1, 0]), ord=1)
            for destination in self._destination
        ])


        # Calculate the total distance to the targets and the total distance to the destination
        #total_distance_to_targets = np.sum(distances_to_targets)
        self.distance_to_destination = np.mean(distances_targets_to_door) 
        self.distance_to_destination = np.mean(distances_targets_to_door) 
        
        self._cached_distances = {
            #'distance_to_goal':total_distance_to_targets,
            'distances_door_to_destination': self.distances_door_to_destination,
            'distances_targets_to_door': self.distances_targets_to_door,
        }
        return self._cached_distances

    
    def _get_info(self):     
        self._cached_distances = {
            'num_targets_left_to_drop_off': self.num_targets_left_to_drop_off,
        }
        return self._cached_distances


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #self._target_picked_up = False
        self._target_picked_up = np.zeros(self.num_targets, dtype=np.int8)
        self._target_dropped_off = np.zeros(self.num_targets, dtype=np.int8)
        self.num_targets_left_to_drop_off = self.num_targets
        

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size_rows-1, size=2, dtype=int)
        
        # nradom row for the target
        self._target_location = []
        for i in range(self.num_targets):
            # Place each target in the last column, from top to bottom
            target_location = [self.size_rows - 1, i]
            self._target_location.append(target_location)

        
        # randomize location for destination
        self._destination = []
        for _ in range(self.num_targets):
            # Randomize location for each destination
            while True:
                target_destination = [random.randint(0, self.size_rows - 2), random.randint(0, self.size_cols - 2)]
                if target_destination not in self._destination:
                    break
            self._destination.append(target_destination)
  
        # Ensure that destination is different from the agent's and targets' locations
        while any(np.array_equal(self._agent_location, dest) for dest in self._destination):
            self._agent_location = self.np_random.integers(0, self.size_rows-1, size=2, dtype=int)

        
        # Assuming targets are not picked up at the start
        self._target_picked_up = [False] * self.num_targets
        self._target_dropped_off = [False] * self.num_targets

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def move_object(self, current_position, direction, boundaries, action):
        """Moves an object, agent or target, within specified boundaries. Allows exception for top right cell.

        Args:
        current_position (np.array): The current position of the object.
        direction (np.array): The direction vector in which to move the object.
        boundaries (tuple): The min and max boundaries for the object's position.

        Returns:
        np.array: The new position of the object after movement.
        """
        last_col = self.size_cols-1
        lat_row = self.size_rows-1
        new_position = current_position
        # Calculate the tentative new position
        tentative_new_position = current_position + direction

        # Can't move right toward the grey area
        if np.array_equal(current_position[0],last_col-1) and  (1<=tentative_new_position[1]<=lat_row-1) and action == 0  :
            # stay in the same position
            new_position = current_position

        elif any(np.array_equal(tentative_new_position, target_loc) for target_loc in self._target_location) and any(self._target_picked_up ):
            return new_position
        
        # Can't move down toward the grey area
        elif np.array_equal(current_position[1],last_col-1) and  (0<=tentative_new_position[0]<=lat_row-1) and action == 1  :
            # stay in the same position
            new_position = current_position

        # Can't move up from the grey area
        elif np.array_equal(current_position[1],last_col) and  (0<=tentative_new_position[0]<=lat_row-1) and action == 3  :
            # stay in the same position
            new_position = current_position

        # Can't move left from the grey area
        elif np.array_equal(current_position[0],last_col) and  (1<=tentative_new_position[1]<=lat_row-1) and action == 2  :
            # stay in the same position
            new_position = current_position
    
        else:
            # Apply clipping for all other moves
            new_position = np.clip(tentative_new_position, boundaries[0], boundaries[1])
 
        return new_position
    

    
    
    def calculate_path_euclidean_distance(self, path):
        total_distance = 0
        for i in range(1, len(path)):
            total_distance += math.sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)
        return total_distance

    
    def progress_queue(self):
        """Progresses the queue of targets to be picked up and dropped off."""
        for i in range(self.num_targets):
            if not self._target_dropped_off[i]:
                door_position = np.array([self.size_rows - 1, 0])  # The position of the door
                direction_to_door = door_position - self._target_location[i]
                step_towards_door = np.sign(direction_to_door)
                self._target_location[i] += step_towards_door
                self._target_location[i] = np.clip(self._target_location[i], [0, 0], [self.size_rows - 1, self.size_cols - 1])
    
    def heuristic(self, a, b):
        """Heuristic function to estimate the distance between two points.

        Args:
        a (np.array): The first point.
        b (np.array): The second point.

        Returns:
        float: The estimated distance between the two points.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_search(self, target, destination, grid_boundary):
        """A* algorithm to find the shortest path from the targets to the destinations.

        Args:
        targets (list): The list of targets to be reached.
        destinations (list): The list of destinations to reach.

        Returns:
        list: The list of actions to be taken by the agent to reach the target.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        target = tuple(target)
        destination = tuple(destination)

        last_col = self.size_cols-1
        lat_row = self.size_rows-1
        
        close_set = set() # Set of cells already evaluated
        came_from = {}  # Keep track of the best path to a node
        gscore = {target: 0}  # Cost from start to the current node
        fscore = {target: self.heuristic(target, destination)}  # Estimated cost from start to goal through the current node
        oheap = []  # Priority queue for nodes to be evaluated

        heapq.heappush(oheap, (fscore[target], target))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == destination:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data[::-1]  # Return reversed path

            close_set.add(current)
            for i, j in directions: 
                neighbor = current[0] + i, current[1] + j            
                tentative_g_score = gscore[current] + 1  # Assume cost from current to neighbor is 1
                if 0 <= neighbor[0] < grid_boundary:
                    if 0 <= neighbor[1] < grid_boundary:
                        if np.array_equal(current[0],last_col) and  (1<=neighbor[1]<=lat_row-1):  # Check if neighbor is an obstacle
                            continue
                        elif np.array_equal(current[1],last_col) and  (0<=neighbor[0]<=lat_row-1): 
                            continue
                        elif np.array_equal(current[1],last_col-1) and  (0<=neighbor[0]<=lat_row-1):
                            continue
                        elif np.array_equal(current[1],last_col-1) and  (1<=neighbor[1]<=lat_row-1):
                            continue
                        elif tuple(neighbor) in map(tuple, self._target_location):
                            continue
                    else:
                        # Neighbor is out of bounds
                        continue
                else:
                    # Neighbor is out of bounds
                    continue
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                
                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, destination)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                    
        return False


    def step(self, action):

        # Gather the current state and information
        observation = self._get_obs()
        info = self._get_info()
        reward = -1  # Default reward for any action that isn't explicitly rewarded

        # Initialize reward and check if the episode has terminated
        boundary = ([0, 0], [self.size_rows - 1, self.size_rows - 1])
        # Check if all targets have been delivered to their respective destinations
        terminated =  all(any(np.array_equal(target_loc, dest) for dest in self._destination) for target_loc in self._target_location)
        # All targets are at their destinations


        # Determine the direction based on the action
        direction = self.get_direction(action)

        #Handle agent movemnt
        if action in [0, 1, 2, 3]:  # Actions for moving the agent
            self._agent_location = self.move_object(self._agent_location, direction, boundary, action)
            

            # Move the target as well if it's picked up
            for i, (target_loc, picked_up, destination) in enumerate(zip(self._target_location, self._target_picked_up, self._destination)):
                if picked_up:
                    self._target_location[i]= self.move_object(self._target_location[i], direction, boundary, action)
                    #path = self.a_star_search(self._target_location[i], destination, self.size_cols)
                    #if path:
                        #distance =  (self.calculate_path_euclidean_distance(path)) + 1 
                        #reward = - distance*0.1
                    #else:
                        #reward = 0
                               

        # Handle picking up a target
        if action == 4:  
            for i, target_loc in enumerate(self._target_location):
                if not self._target_dropped_off[i]:
                    if np.array_equal(self._agent_location, target_loc) and not self._target_picked_up[i]:
                        self._target_picked_up[i] = True
                        #reward = 1  # Reward for picking up a target
                        break  # Ensure to exit the loop after handling the pick-up
                    #else:
                        #reward = -3  # Penalty got picking up at a wrong location
                        #print("Pcking at a wrong location")
                #else:
                    #reward = -2  # Penalty for trying to pick up a target that has already been dropped off
                    #print("Picking up a target that has already been dropped off")

        # Handle dropping off a target 
        if action == 5:
            for i, (target_loc, picked_up) in enumerate(zip(self._target_location, self._target_picked_up)):
                # Check if the target has been picked up and is at the correct location
                if picked_up and np.array_equal(target_loc, self._agent_location):
                    # Check if the location is one of the destinations
                    if any(np.array_equal(target_loc, dest) for dest in self._destination):
                        # Now check if the target has not been dropped off yet
                        if not self._target_dropped_off[i]:
                            self._target_picked_up[i] = False
                            self._target_dropped_off[i] = True
                            reward = 20  # Reward for correctly dropping off the target
                            print("A target has been dropped off at the destination")
                            #print("Single target dropped off at the destination")
                            self.num_targets_left_to_drop_off -= 1
                            self.progress_queue()
                            break  # Ensure to exit the loop after handling the drop-off
                        #else:
                            #reward = -1  # Penalty for trying to drop off a target that has already been dropped off
                            #self._target_picked_up[i] = False
                            
                    #else:
                        #reward = -1  # Penalty for dropping off a target at the wrong location
                        #self._target_picked_up[i] = False
                        
                #else:
                    #reward = -1  # Penalty for trying to drop off a target that hasn't been picked up or isn't at the correct location
                    #self._target_picked_up[i] = False
                    

        if terminated: 
            reward = self.num_targets*30  # Reward for dropping off the last target at the destination
            print("------------------------------------------>Success")
        
        
        #print("Reward: ", reward)



 

        # Render if in human mode
        if self.render_mode == "human":
            self._render_frame()

        observation, info = self._get_obs(), self._get_info() 
        return observation, reward, terminated, False, info
    



        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        

        canvas = pygame.Surface((self.screen_width,self.screen_height))
        canvas.fill((2, 19, 46)) # The size of a single grid square in pixels


        # Draw grid lines
        #pix_square_size_width = self.window_width //self.size_cols
        pix_square_size_height = self.screen_height //self.size_rows
        for x in range(self.number_cells):
            pygame.draw.line(canvas, (29, 191, 219), (0, pix_square_size_height * x), (self.screen_height, pix_square_size_height  * x), 3)
            pygame.draw.line(canvas, (29, 191, 219), (pix_square_size_height * x, 0), (pix_square_size_height * x, self.screen_width), 3)

        # Additional lines at the extremities of the grid
        pygame.draw.line(canvas, (29, 191, 219), (self.screen_width, self.screen_width // self.number_cells), (self.screen_width, self.screen_width), 3)
        pygame.draw.line(canvas, (29, 191, 219), (0, self.screen_width), (self.screen_width, self.screen_width), 6)

        # Fill the last row and column with a distinct color
        last_row_start_y = (self.size_rows - 1) * pix_square_size_height
        last_col_start_x = (self.size_cols - 1) * pix_square_size_height
        non_accessible_color = (176, 188, 209)  # Example: Red color for non-accessible areas

        # Fill the last row
        pygame.draw.rect(
            canvas,
            non_accessible_color,
            (0, last_row_start_y, self.screen_width, pix_square_size_height)
        )

        # Fill the last column
        pygame.draw.rect(
            canvas,
            non_accessible_color,
            (last_col_start_x, 0, pix_square_size_height, self.screen_height)
        )

        #circle drawing
        radius = 40
        #pygame.draw.circle(self.screen, (245, 242, 245), self._agent_location[0]*pix_square_size_height, 
                                                         #(self._agent_location[1]*pix_square_size_height),radius)
        # Calculate the center of the grid cell for the agent
        agent_center_x = (self._agent_location[0]+0.5)*pix_square_size_height
        agent_center_y = (self._agent_location[1]+0.5)*pix_square_size_height

        # Draw a circle representing the agent at the center of the grid cell
        pygame.draw.circle(
            canvas,
            (245, 242, 245),
            (agent_center_x, agent_center_y),
            radius
        )

        # Draw each destination
        for target_destination in self._destination:
            target_center_x = (target_destination[1] + 0.5) * pix_square_size_height
            target_center_y = (target_destination[0] + 0.5) * pix_square_size_height
            pygame.draw.rect(
                canvas,
                (37, 95, 219),  # Target color
                (int(target_center_y) - 35, int(target_center_x) - 35, 70, 70),  # Adjust position and size as needed
                border_radius=10
            )

        # Draw each target
        for target_location in self._target_location:
            target_center_x = (target_location[1] + 0.5) * pix_square_size_height
            target_center_y = (target_location[0] + 0.5) * pix_square_size_height
            pygame.draw.rect(
                canvas,
                (138, 73, 131),  # Target color
                (int(target_center_y) - 32, int(target_center_x) - 32, 65, 65),  # Adjust position and size as needed
                border_radius=10
            )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
