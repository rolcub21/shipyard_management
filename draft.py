import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.locals import *
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", size=6):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
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
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
''
import gymnasium as gym
from gymnasium import spaces
import random 
import pygame
from pygame.locals import *
import numpy as np



class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="rgb_array", size=6):
        self.size_rows = size  # The size of the square grid
        self.size_cols = size + 1
        #self.window_size = 512  # The size of the PyGame window
        self.window_width = (self.size_cols*100)
        self.window_height = self.size_rows*100
        self.screen_width = size*100
        self.screen_height = size*100
        self.number_cells = size
        self.distance_to_target = 0.1
        self.direction =  np.array([0, 0]),

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size_rows -1 , shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size_rows -1 , shape=(2,), dtype=int),
                "target_picked_up": spaces.Discrete(2),
                "destination": spaces.Box(0, self.size_rows - 1, shape=(2,), dtype=int),
               
                #"agent" : spaces.Box(low=np.array([0, 0]), high=np.array([self.size_rows, self.size_rows-1]), shape=(2,), dtype=int),
                #"target" : spaces.Box(low=np.array([0, 0]), high=np.array([self.size_rows , self.size_rows-2]), shape=(2,), dtype=int),
            }
        )


        # We have 9 actions, corresponding to "right", "up", "left", "down", "right", same actions with the target and a null action
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
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        return action_to_direction.get(action)

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "target_picked_up": self._target_picked_up, "destination": self._destination}
    

    
    def _get_info(self):
        # Calculate and cache distances only if they have not been calculated before or if positions have changed
        if not hasattr(self, '_cached_distances') or \
        not np.array_equal(self._cached_distances['prev_agent_loc'], self._agent_location) or \
        not np.array_equal(self._cached_distances['prev_target_loc'], self._target_location):
            
            self._cached_distances = {
                'distance_to_goal': np.linalg.norm(self._target_location - self._destination, ord=1),
                'distance_to_target': np.linalg.norm(self._agent_location - self._target_location, ord=1),
                'prev_agent_loc': np.copy(self._agent_location),
                'prev_target_loc': np.copy(self._target_location)
            }
    
        return self._cached_distances


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._target_picked_up = False
        

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size_rows-1, size=2, dtype=int)
        
        # nradom row for the target
        self._target_location = np.copy(self._agent_location)
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size_rows-1, size=2, dtype=int)
        
        # randomize location for destination
        self._destination = self.np_random.integers(0, self.size_rows-1, size=2, dtype=int)    

        #check that destination is different from the agent's and target's locations
        while np.array_equal(self._destination, self._agent_location) or np.array_equal(self._destination, self._target_location):
            self._destination = self.np_random.integers(0, self.size_rows - 1, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def move_object(self, current_position, direction, boundaries):
        """Moves an object, agent or target, within specified boundaries

        Args:
        current_position (np.array): The current position of the object.
        direction (np.array): The direction vector in which to move the object.
        boundaries (tuple): The min and max boundaries for the object's position.

        Returns:
        np.array: The new position of the object after movement.
        """
        new_position = np.clip(current_position + direction, boundaries[0], boundaries[1])
        return new_position



    def step(self, action):

        # Gather the current state and information
        observation = self._get_obs()
        info = self._get_info()
        reward = 0
        
        # Initialize reward and check if the episode has terminated
        
        terminated = np.array_equal(self._target_location, self._destination)
        boundary = ([0, 0], [self.size_rows - 1, self.size_rows - 1])

        # Determine the direction based on the action
        direction = self.get_direction(action)

        #Handle agent movemnt
        if action in [0, 1, 2, 3]:  # Actions for moving the agent
            self._agent_location = self.move_object(self._agent_location, direction, boundary)
            reward = -1
            if self._target_picked_up:  # Move the target as well if it's picked up
                self._target_location = self.move_object(self._target_location, direction, boundary)
                reward = 1


        #Handle picking up the block
        if action == 4:
            if np.array_equal(self._agent_location, self._target_location) and not self._target_picked_up:
                self._target_picked_up = True
                reward = 3
                print("Picked action 4")
            else:
                reward = -5  # Optionally, penalize unnecessary pick-up attempts

        #Handle dropping off the block
        if action == 5:
                if np.array_equal(self._agent_location, self._target_location):
                    self._target_picked_up = False
                    print("Picked action 5")
                    if terminated:
                        print("Success: Target dropped off at destination. Terminating episode.")
                        reward = 10
                    else:
                        reward = -2
                else:

                    reward = -2  # Optionally, penalize unnecessary drop-off attempts

        # Calculate distance-based reward or penalty
                    
           
        #if not self._target_picked_up:
            # Penalize based on distance to target when the target is not picked up
            #distance_penalty = info['distance_to_target']*0.5
            #reward = -distance_penalty 
        #if self._target_picked_up:
            # Penalize based on distance to destination when the target is picked up
            #distance_penalty = info['distance_to_goal']*0.5  # Use distance to goal here
            #reward = -distance_penalty 

        #reward 

        # Render if in human mode
        if self.render_mode == "human":
            self._render_frame()

        return  observation, reward, terminated, False, info

        
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
        

        #screen = pygame.display.set_mode((self.window_width, self.window_height))
        surface1 = pygame.Surface((self.screen_width, self.screen_height))
        surface2 = pygame.Surface((self.window_width , self.screen_height))
        canvas = pygame.Surface((self.screen_width,self.screen_height))
        # Load images
        #bg_image = pygame.image.load('bg_02_h.png')
 

        # Transform and scale images
        #bg_image = pygame.transform.scale(bg_image, (self.screen_width, self.screen_height))
        #truck_image = pygame.transform.scale(truck_image, ((self.screen_width // self.number_cells) * 7 // 10,) * 2)

        # Fill surfaces
        #surface1.blit(bg_image, (0, 0))
        canvas.fill((2, 19, 46)) # The size of a single grid square in pixels

        # Fill background with surfaces
        #self.screen.blit(surface1, (0, 0))
        #self.screen.blit(surface2, (self.screen_width, 0))

        # Draw grid lines
        #pix_square_size_width = self.window_width //self.size_cols
        pix_square_size_height = self.screen_height //self.size_rows
        for x in range(self.number_cells):
            pygame.draw.line(canvas, (29, 191, 219), (0, pix_square_size_height * x), (self.screen_height, pix_square_size_height  * x), 3)
            pygame.draw.line(canvas, (29, 191, 219), (pix_square_size_height * x, 0), (pix_square_size_height * x, self.screen_width), 3)

        # Additional lines at the extremities of the grid
        pygame.draw.line(canvas, (29, 191, 219), (self.screen_width, self.screen_width // self.number_cells), (self.screen_width, self.screen_width), 3)
        pygame.draw.line(canvas, (29, 191, 219), (0, self.screen_width), (self.screen_width, self.screen_width), 6)

        # Place agent image (truck)
        #screen.blit(truck_image, (int((screen_width / number_cells) * 2.2), int((screen_width / number_cells) * 2.1)))

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

        #blocks
        pygame.draw.rect(
            canvas,
            (138, 73, 131),
            (self._target_location[0]*pix_square_size_height+14, self._target_location[1]*pix_square_size_height+14, 70, 70),
            0,
            border_radius=10,
        )

        # Highlight the destination location
        # You can choose a different color or shape to distinguish the destination
        destination_color = (0, 255, 0)  # Green color for destination
        pygame.draw.rect(
            canvas,
            destination_color,
            (self._destination[0] * pix_square_size_height + 14, self._destination[1] * pix_square_size_height + 14, 70, 70),
            0,
            border_radius=10,
        ) 


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            #pygame.display.flip()
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            
            image_data = pygame.surfarray.array3d(canvas)
            return np.transpose(image_data, axes=(1, 0, 2))
            #return np.transpose(
                #np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            #)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()










