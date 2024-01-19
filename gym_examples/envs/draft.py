import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.locals import *
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", size=6):
        self.size = size  # The size of the square grid
        #self.window_size = 512  # The size of the PyGame window
        self.window_width = (size*100)+100
        self.window_height = size*100
        self.screen_width = size*100
        self.screen_height = size*100
        self.number_cells = size

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "block": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        #self._action_to_direction = {
            #0: np.array([1, 0]),
            #1: np.array([0, 1]),
            #2: np.array([-1, 0]),
            #3: np.array([0, -1]),
        #}

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
        return {"agent": self._agent_location, "block": self._target_location}

    #def _get_info(self):
        #return {
            #"distance": np.linalg.norm(
                #self._agent_location - self._target_location, ord=1
            #)
        #}
    
    def get_info(self):
        distance_to_target = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        goal_reached = self.is_rectangle_in_last_column() is False
        #steps_taken = self._step_counter  # Assuming you have a step counter in your environment
    
        return {
            "distance_to_target": distance_to_target,
            "goal_reached": goal_reached,
            #"steps_taken": steps_taken,
            # Other relevant information...
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

    
    def is_rectangle_in_last_column(self):
        return self._target_location[0] == self.size - 1

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in

        if action == 0: #moving right
            self.circle_position[0] += 1

        elif action == 1: #moving left
            self.circle_position[0] -= 1

        elif action == 2: #moving up
            self.circle_position[1] += 1

        elif action == 3: #moving down
            self.circle_position[1] -= 1  

        elif action == 4: #pickung a block
            if self.circle_position == self.rectangle_position:
                self.rectangle_position = self.circle_position.copy()

        elif action == 5: #moving right with a block
            if self.circle_position == self.rectangle_position:
                self.circle_position[0] += 1 
                self.rectangle_position[0] += 1 
            else:
                pass

        elif action == 6: #moving left with a block
            if self.circle_position == self.rectangle_position:
                self.circle_position[0] -= 1 
                self.rectangle_position[0] -= 1 
            else:
                pass   
                
        elif action == 7: #moving up with a block
            if self.circle_position == self.rectangle_position:
                self.circle_position[1] += 1 
                self.rectangle_position[1] += 1 
            else:
                pass   

        elif action == 8: #moving down with a block
            if self.circle_position == self.rectangle_position:
                self.circle_position[1] -= 1 
                self.rectangle_position[1] -= 1 
            else:
                pass  



        # We use `np.clip` to make sure we don't leave the grid   
        self._circle_location[1] = np.clip(self._agent_location[1] + action[1], 0, self.window_height - 1) 
        self._circle_location[0] = np.clip(self._agent_location[0] + action[0], 0, self.window_width - 1)  # For x-axis
        
        #restrict right action in the right side of the grid, except for the door
        if self._agent_location[0] == self.size-2 and 0 < self._agent_location[1] < self.size - 1 and action in [0,5]: 
            pass

        # An episode is done iff the agent has reached the target
        terminated = not self.is_rectangle_in_last_column()
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info





    def ship_block(self):
        pass

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

        screen = pygame.display.set_mode((self.window_width, self.window_height))
        surface1 = pygame.Surface((self.screen_width, self.screen_height))
        surface2 = pygame.Surface((self.window_width - self.screen_width, self.screen_height))

        # Load images
        bg_image = pygame.image.load('bg_02_h.png')
        truck_image = pygame.image.load('truck.png')

        # Transform and scale images
        #bg_image = pygame.transform.scale(bg_image, (screen_width, screen_height))
        truck_image = pygame.transform.scale(truck_image, ((self.screen_width // self.number_cells) * 7 // 10,) * 2)

        # Fill surfaces
        surface1.blit(bg_image, (0, 0))
        surface2.fill((140, 143, 194)) # The size of a single grid square in pixels

        # Fill background with surfaces
        screen.blit(surface1, (0, 0))
        screen.blit(surface2, (self.screen_width, 0))

        # Draw grid lines
        pix_square_size = self.screen_width //self. number_cells
        for x in range(self.number_cells):
            pygame.draw.line(screen, (29, 191, 219), (0, pix_square_size * x), (self.screen_width, pix_square_size * x), 3)
            pygame.draw.line(screen, (29, 191, 219), (pix_square_size * x, 0), (pix_square_size * x, self.screen_height), 3)

        # Additional lines at the extremities of the grid
        pygame.draw.line(screen, (29, 191, 219), (self.screen_width, self.screen_width // self.number_cells), (self.screen_width, self.screen_width), 3)
        pygame.draw.line(screen, (29, 191, 219), (0, self.screen_width), (self.screen_width, self.screen_width), 6)

        # Place agent image (truck)
        #screen.blit(truck_image, (int((screen_width / number_cells) * 2.2), int((screen_width / number_cells) * 2.1)))

        #circle drawing
        pygame.draw.circle(screen, (245, 242, 245), ((self.screen_width / self.number_cells) * 2.5, (self.screen_width / self.number_cells) * 2.5), pix_square_size*0.4 )

        #blocks
        pygame.draw.rect(
            screen,
            (138, 73, 131),
            [(self.screen_width / self.number_cells) * 6.15, (self.screen_width / self.number_cells) * 5.15, pix_square_size*0.7, pix_square_size*0.7],
            0,
            border_radius=10,
        )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            pygame.display.flip()
            #self.window.blit(canvas, canvas.get_rect())
            #pygame.event.pump()
            #pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
