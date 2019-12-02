# multi-agent gridworld environment
# agents must cooperate to move a box to a goal position

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import contextlib
with contextlib.redirect_stdout(None): # removes welcome message
    import pygame


class BoxCarryEnv(gym.Env):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4
    GRAB = 5
    RELEASE = 6

    metadata = {'render.modes': ['human', 'rgb_array']}
    
    num_agents = 2
    num_grabbers_needed = 2

    assert num_agents <= 4 # max one agent per box side
    agents_start = [[0,i] for i in range(num_agents)] # arbitrary

    action_space = spaces.Discrete(7) # 4 cardinal directions, grab, release, and no-op

    def __init__(self, field_size=96, agent_size=32, mode="rgb_array"):
        self.seed()

        assert field_size % agent_size == 0

        self.field_size = field_size
        self.agent_size = agent_size
        self.grid_size = int(field_size/agent_size)
        
        # arbitrary
        self.box_start = [self.grid_size//2, self.grid_size//2]
        self.box_goal = [self.grid_size-1, self.grid_size-1]

        self.mode = mode
        if self.mode=="human":
            pygame.display.init()
            self.surface = pygame.display.set_mode([field_size, field_size])

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.field_size, self.field_size, 3), dtype=np.uint8)

        channel = np.ones((self.agent_size, self.agent_size), dtype=np.uint8)
        self.blue_square = np.stack([channel*0, channel*0, channel*255], axis=-1)
        self.red_square = np.stack([channel*255, channel*0, channel*0], axis=-1)
        self.green_square = np.stack([channel*0, channel*255, channel*0], axis=-1)

        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        assert len(actions) == BoxCarryEnv.num_agents and all([BoxCarryEnv.action_space.contains(action) for action in actions])
            
        grabbers = [i for (i,x) in enumerate(self.agents_grabbing) if x == 1]
        non_grabbers = [i for i in range(BoxCarryEnv.num_agents) if i not in grabbers]

        # if sufficient number of agents are grabbing the box
        if len(grabbers) >= BoxCarryEnv.num_grabbers_needed:
            new_positions = self.agents_pos.copy()
            box_delta = np.array([0., 0.])
            
            for agent in grabbers:
                new_positions[agent] = self.agent_step(agent, actions[agent])
                box_delta += (new_positions[agent] - self.agents_pos[agent])/len(grabbers) # all grabbers must cooperate

            # if the box was moved in the same direction by all grabbers
            if 1 in box_delta or -1 in box_delta:
                new_box_pos = self.box_pos + box_delta.astype(np.int32)
                if self.is_in_bounds(new_box_pos) and self.is_clear(new_box_pos):
                    self.box_pos = new_box_pos
                    self.agents_pos = new_positions
        else:
            # agents grabbing the box will not move, but they might still release
            for agent in grabbers:
                self.agent_step(agent, actions[agent])

        # non-grabbing agents can potentially move
        # arbiitrary ordering determines collision resolution
        for agent in non_grabbers:
            self.agents_pos[agent] = self.agent_step(agent, actions[agent])
            
        observation = self.render()
        reward = self.get_reward()
        done = all(self.box_pos == self.box_goal)
        info = {}
        
        return observation, reward, done, info


    def is_pos_good(self):
        for i,pos in enumerate(self.agents_pos):
            if all(pos == self.box_pos):
                print(pos, self.box_pos)
                return False
        return True


    def get_reward(self):
        x_dist = self.box_pos[1] - self.box_goal[1]
        y_dist = self.box_pos[0] - self.box_goal[0]
        return -np.sqrt(x_dist**2 + y_dist**2)
    

    # returns would-be new position if agent took action
    def agent_step(self, agent, action):
        pos = self.agents_pos[agent].copy()

        # if staying, grabbing, or releasing, no need to check if current position is valid
        if action == BoxCarryEnv.STAY:
            return pos
        if action == BoxCarryEnv.RELEASE:
            self.agents_grabbing[agent] = 0
            return pos
        if action == BoxCarryEnv.GRAB and self.is_near_box(agent):
            self.agents_grabbing[agent] = 1
            return pos
        
        if action == BoxCarryEnv.LEFT:
            pos[1] -= 1
        elif action == BoxCarryEnv.RIGHT:
            pos[1] += 1
        elif action == BoxCarryEnv.UP:
            pos[0] -= 1
        elif action == BoxCarryEnv.DOWN:
            pos[0] += 1
            
        if self.is_valid(agent, pos):
            return pos
        return self.agents_pos[agent]


    # returns true if agent is only one step away from box position
    def is_near_box(self, agent):
        agent_box_diff = abs(self.agents_pos[agent] - self.box_pos)
        return agent_box_diff.sum() == 1

    # is new position within grid bounds
    def is_in_bounds(self, pos):
        return all([coord >= 0 and coord < self.grid_size for coord in pos]) 

    # is new position occupied by an agent
    def is_occupied(self, pos):
        return any([all(self.agents_pos[i] == pos) for i in range(BoxCarryEnv.num_agents)]) 

    # is new position blocked by un-grabbed box
    def is_blocked(self, pos, agent):
        return self.agents_grabbing[agent] == 0 and all(self.box_pos == pos)

    # is new box position clear of non_grabbers
    def is_clear(self, pos):
        return not any([self.agents_grabbing[i] == 0 and all(self.agents_pos[i] == pos) for i in range(BoxCarryEnv.num_agents)]) 

    def is_valid(self, agent, pos):
        return self.is_in_bounds(pos) and not self.is_occupied(pos) and not self.is_blocked(pos, agent)


    def reset(self):
        self.agents_pos = np.array(BoxCarryEnv.agents_start)
        self.box_pos = np.array(self.box_start)
        self.agents_grabbing = [0]*BoxCarryEnv.num_agents
        return self.render()
        

    def insert_square(self, arr, pos, color):
        if color == "blue":
            square = self.blue_square
        elif color == "red":
            square = self.red_square
        else:
            square = self.green_square

        scale = self.agent_size
        arr[pos[0]*scale : (pos[0]+1)*scale, pos[1]*scale : (pos[1]+1)*scale] = square


    def render(self):
        arr = np.ones(self.observation_space.shape, dtype=np.uint8)*255 # white background

        # agents
        for pos, grabbing in zip(self.agents_pos, self.agents_grabbing):
            if grabbing == 0:
                self.insert_square(arr, pos, "blue")
            else:
                self.insert_square(arr, pos, "green")

        # box
        self.insert_square(arr, self.box_pos, "red")

        if self.mode == 'rgb_array':
            return arr    

        elif self.mode == 'human':
            pygame.surfarray.blit_array(self.surface, arr)
            pygame.display.flip()
            time.sleep(.001)
            return arr
            
        else:
            super(MyEnv, self).render() # just raise an exception


# running this file (not importing) tests and displays
if __name__ == "__main__":
    env = BoxCarryEnv(mode="human")
    for _ in range(100000):
        env.step([env.action_space.sample() for _ in range(env.num_agents)])
