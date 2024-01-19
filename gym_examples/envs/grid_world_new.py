import gymnasium as gym
from gymnasium import spaces
import random 
import pygame
from pygame.locals import *
import numpy as np



class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", size=6):
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
        return {"agent": self._agent_location, "target": self._target_location, "target_picked_up": self.target_picked_up, "destination": self._destination}
    
    def _get_info(self):
        #barrier_hit = self.is_barrier(self.last_action) if hasattr(self, 'last_action') else False
    
        return {
            "distance_to_goal": np.linalg.norm(np.array(self._destination) - np.array(self._target_location), ord=1),
            "distance_to_target": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            #"barrier_hit": barrier_hit
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.target_picked_up = False
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size_rows-1, size=2, dtype=int)
        
        # nradom row for the target
        random_row_target = random.randint(0, self.size_rows-1)
        #self._agent_location fixed position
        self._target_location = np.array([self.size_rows-1, random_row_target])
        self._destination = ([3, 3])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
   
    def is_barrier(self, action):
        if (self._agent_location[0] == self.size_rows-1 and action in [0, 4] and 0 < self._agent_location[1] < self.size_rows) or (self._agent_location[0] == self.size_rows and action in [2, 6] and 0 < self._agent_location[1] < self.size_rows):
            pass

    def step(self, action):
        direction = self.get_direction(action)
        if (self._agent_location[0] == self.size_rows-1 and action in [0, 4] and 0 < self._agent_location[1] < self.size_rows) or (self._agent_location[0] == self.size_rows and action in [2, 6] and 0 < self._agent_location[1] < self.size_rows):
            pass
        elif action == 0: #moving right
            if self.target_picked_up is False:
                self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            else:
                 self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
                 self._target_location = np.clip(self._target_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            print(self._agent_location)
            print(self._target_location)

        elif action == 1: #moving down
            if self.target_picked_up is False: 
                self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            else:
                 self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
                 self._target_location = np.clip(self._target_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            print(self._agent_location)
            print(self._target_location)

        elif action == 2: #moving left
            if self.target_picked_up is False: 
                self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            else:
                self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
                self._target_location = np.clip(self._target_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            print(self._agent_location)
            print(self._target_location)
  
        elif action == 3: #moving up
            if self.target_picked_up is False: 
                self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            else:
                 self._agent_location = np.clip(self._agent_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
                 self._target_location = np.clip(self._target_location + direction, [0, 0], [self.size_rows-1, self.size_rows-1])
            print(self._agent_location)
            print(self._target_location)

        elif action == 4: #picking up a block
            if np.array_equal(self._agent_location, self._target_location):
                if self.target_picked_up is False:
                    self.target_picked_up = True
                else: 
                    reward = -2
            else:
                reward = -1
                print(self._agent_location)
                print(self._target_location)

        elif action == 5: #dropping off a block
            if np.array_equal(self._agent_location, self._target_location):
                if self.target_picked_up is True:
                    self.target_picked_up = False
                else:
                    reward = -2
            else:
                reward = -1
                print(self._agent_location)
                print(self._target_location)
                



        
        #terminated = self.is_rectangle_in_last_column()
        observation = self._get_obs()
        terminated = np.array_equal(self._target_location, self._destination)
        info = self._get_info()
        distance = info['distance_to_goal']
        #reward =1 if terminated else 0
        if terminated:
            reward = 5
        else:
            if (np.array_equal(self._agent_location, self._target_location)):
                reward = 1
            else:
                distance_penalty = (info['distance_to_goal'] + info['distance_to_goal']) * 0.2
                reward = - distance_penalty
        #reward = 1 if terminated else -distance  # Binary sparse rewards
        


        if self.render_mode == "human":
            self._render_frame()

        return  observation, reward, terminated, False, info
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        
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
        surface2 = pygame.Surface((self.window_width - self.screen_width, self.screen_height))

        # Load images
        bg_image = pygame.image.load('bg_02_h.png')
 

        # Transform and scale images
        #bg_image = pygame.transform.scale(bg_image, (self.screen_width, self.screen_height))
        #truck_image = pygame.transform.scale(truck_image, ((self.screen_width // self.number_cells) * 7 // 10,) * 2)

        # Fill surfaces
        surface1.blit(bg_image, (0, 0))
        surface2.fill((140, 143, 194)) # The size of a single grid square in pixels

        # Fill background with surfaces
        self.screen.blit(surface1, (0, 0))
        self.screen.blit(surface2, (self.screen_width, 0))

        # Draw grid lines
        #pix_square_size_width = self.window_width //self.size_cols
        pix_square_size_height = self.screen_height //self.size_rows
        for x in range(self.number_cells):
            pygame.draw.line(self.screen, (29, 191, 219), (0, pix_square_size_height * x), (self.screen_height, pix_square_size_height  * x), 3)
            pygame.draw.line(self.screen, (29, 191, 219), (pix_square_size_height * x, 0), (pix_square_size_height * x, self.screen_width), 3)

        # Additional lines at the extremities of the grid
        pygame.draw.line(self.screen, (29, 191, 219), (self.screen_width, self.screen_width // self.number_cells), (self.screen_width, self.screen_width), 3)
        pygame.draw.line(self.screen, (29, 191, 219), (0, self.screen_width), (self.screen_width, self.screen_width), 6)

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
            self.screen,
            (245, 242, 245),
            (agent_center_x, agent_center_y),
            radius
        )

        #blocks
        pygame.draw.rect(
            self.screen,
            (138, 73, 131),
            (self._target_location[0]*pix_square_size_height+14, self._target_location[1]*pix_square_size_height+14, 70, 70),
            0,
            border_radius=10,
        )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            #pygame.display.flip()
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()





