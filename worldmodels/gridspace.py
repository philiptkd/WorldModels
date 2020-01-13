from gym.spaces.space import Space
import numpy as np

class GridSpace(Space):
    def __init__(self, side_length, num_agents):
        assert isinstance(side_length, int)
        assert isinstance(num_agents, int)

        self.side_length = side_length
        self.size = side_length**2

        assert num_agents < self.size
        self.num_agents = num_agents

        super(GridSpace, self).__init__((self.size,), np.int8)


    def sample(self, box_pos=None):
        num_to_sample = self.num_agents

        if box_pos is not None:
            assert len(box_pos) == 2
            assert all([isinstance(x, int) and x < self.side_length for x in box_pos])
            
            box_pos = box_pos[0]*self.side_length + box_pos[1]
        else:
            num_to_sample += 1

        # sample indexes of agents and possibly the box
        idxs = self.np_random.choice(self.size, size=num_to_sample, replace=False)
        
        obs = np.zeros(self.size)
        obs[agent_idxs] = 1

        if box_pos is None:
            box_pos = idxs[-1]
        obs[box_pos] = 2

        return obs


    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return ((x==0) | (x==1) | (x==2)).all()


    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()


    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]


    def __repr__(self):
        return "GridSpace(side_length={}, num_agents={})".format(
                self.side_length, self.num_agents)


    def __eq__(self, other):
        return isinstance(other, GridSpace) and self.side_length == other.side_length \
                and self.num_agents == other.num_agents and self.box_pos == other.box_pos
