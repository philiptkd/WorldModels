# multi-agent gridworld environment
# agents must cooperate to move a box to a goal position

import gym
from gym import spaces
import numpy as np
import time

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
    assert num_agents <= 4 # max one agent per box side
    agents_start = [[0,i] for i in range(num_agents)] # arbitrary
    
    action_space = spaces.Discrete(7) # 4 cardinal directions, grab, release, and no-op

    def __init__(self, field_size=96, agent_size=16, mode="rgb_array"):
        if mode == "human":
            import contextlib
            with contextlib.redirect_stdout(None): # removes welcome message
                import pygame

        assert field_size % agent_size == 0

        self.field_size = field_size
        self.agent_size = agent_size
        self.grid_size = int(field_size/agent_size)
        
        # arbitrary
        self.box_start = [self.grid_size//2, self.grid_size//2]
        self.box_goal = [self.grid_size-1, self.grid_size-1]

        self.mode = mode
        if mode=="human":
            pygame.display.init()
            self.surface = pygame.display.set_mode([field_size, field_size])

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.field_size, self.field_size, 3), dtype=np.uint8)

        channel = np.ones((self.agent_size, self.agent_size), dtype=np.uint8)
        self.blue_square = np.stack([channel*0, channel*0, channel*255], axis=-1)
        self.red_square = np.stack([channel*255, channel*0, channel*0], axis=-1)
        self.green_square = np.stack([channel*0, channel*255, channel*0], axis=-1)

        self.reset()


    def step(self, actions):
        try:
            assert len(actions) == BoxCarryEnv.num_agents and all([BoxCarryEnv.action_space.contains(action) for action in actions])
        except AssertionError:
            print(actions)
            raise AssertionError

        # box can only be moved by all agents
        if all([grabbing == 1 for grabbing in self.agents_grabbing]):
            new_positions = self.agents_pos.copy()
            box_delta = np.array([0., 0.])
            
            for i, action in enumerate(actions):
                new_positions[i] = self.agent_step(i, actions[i])
                box_delta += (new_positions[i] - self.agents_pos[i])/BoxCarryEnv.num_agents

            if all(box_delta.astype(int) == box_delta): # if the box was moved in the same direction by all agents
                new_box_pos = self.box_pos + box_delta.astype(np.int32)
                if self.is_in_bounds(new_box_pos):
                    self.box_pos = new_box_pos
                    self.agents_pos = new_positions

        else:
            grabbers = [i for (i,x) in enumerate(self.agents_grabbing) if x == 1]
            non_grabbers = [i for i in range(BoxCarryEnv.num_agents) if i not in grabbers]

            # agents grabbing the box will not move, but they might still release
            for agent in grabbers:
                self.agent_step(agent, actions[agent])

            # non-grabbing agents can potentially move
            # arbiitrary ordering determines collision resolution
            for agent in non_grabbers:
                self.agents_pos[agent] = self.agent_step(agent, actions[agent])

        observation = self.render(mode=self.mode)
        reward = self.get_reward()
        done = all(self.box_pos == self.box_goal)
        info = {}
        
        return observation, reward, done, info


    def get_reward(self):
        x_dist = self.box_pos[1] - self.box_goal[1]
        y_dist = self.box_pos[0] - self.box_goal[0]
        return -np.sqrt(x_dist**2 + y_dist**2)
    

    # returns would-be new position if agent took action
    def agent_step(self, agent, action):
        pos = self.agents_pos[agent].copy() # (x,y) tuple

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
            pos[0] += 1
        elif action == BoxCarryEnv.DOWN:
            pos[0] -= 1
            
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

    # is new position occupied by annother agent
    def is_occupied(self, pos, agent):
        return any([i != agent and all(self.agents_pos[i] == pos) for i in range(BoxCarryEnv.num_agents)]) 

    # is new position blocked by un-grabbed box
    def is_blocked(self, pos, agent):
        return self.agents_grabbing[agent] == 0 and all(self.box_pos == pos)

    def is_valid(self, agent, pos):
        return self.is_in_bounds(pos) and not self.is_occupied(pos, agent) and not self.is_blocked(pos, agent)


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


    def render(self, mode='rgb_array'):
        arr = np.ones(self.observation_space.shape, dtype=np.uint8)*255 # white background

        # agents
        for pos, grabbing in zip(self.agents_pos, self.agents_grabbing):
            if grabbing == 0:
                self.insert_square(arr, pos, "blue")
            else:
                self.insert_square(arr, pos, "green")

        # box
        self.insert_square(arr, self.box_pos, "red")

        if mode == 'rgb_array':
            return arr    

        elif mode == 'human':
            pygame.surfarray.blit_array(self.surface, arr)
            pygame.display.flip()
            time.sleep(.001)
            
        else:
            super(MyEnv, self).render(mode=mode) # just raise an exception


# running this file (not importing) tests and displays
if __name__ == "__main__":
    env = BoxCarryEnv(mode="human")
    for _ in range(100000):
        env.step([env.action_space.sample() for _ in range(env.num_agents)])
