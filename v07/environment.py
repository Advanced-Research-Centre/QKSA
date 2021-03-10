import gym


# https://github.com/openai/gym/blob/master/gym/envs/toy_text/guessing_game.py


env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()