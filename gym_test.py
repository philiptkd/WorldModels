import multiworld
import gym

#env = gym.make('Ant-v2')

multiworld.register_all_envs()
#env = gym.make('SawyerPushNIPS-v0')
#env = gym.make('GoalGridworld-v0')
#env = gym.make('Point2DEnv-Image-v0')

from multiworld.core.image_env import ImageEnv
from multiworld.envs.pygame.point2d import Point2DEnv
env = Point2DEnv(
    images_are_rgb=True,
    render_onscreen=False,
    show_goal=False,
    ball_radius=0.5,
    render_size=10,
    boundary_dist=9,
)
env = ImageEnv(env, imsize=env.render_size, transpose=True)

env.reset()

for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())
    
#env.close()
