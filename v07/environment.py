import gym

# print(gym.envs.registry.all()) # Available environments in gym

# https://github.com/openai/gym/blob/master/gym/envs/toy_text/guessing_game.py
'''
The object of the game is to guess within 1% of the randomly chosen number within 200 time steps
After each step the agent is provided with one of four possible observations which indicate where the guess is in relation to the randomly chosen number
  0 - No guess yet submitted (only after reset)
  1 - Guess is lower than the target
  2 - Guess is equal to the target
  3 - Guess is higher than the target
The rewards are:
  0 if the agent's guess is outside of 1% of the target
  1 if the agent's guess is inside 1% of the target
The episode terminates after the agent guesses within 1% of the target or 200 steps have been taken
The agent will need to use a memory of previously submitted actions and observations in order to efficiently explore the available actions
'''
env = gym.make('GuessingGame-v0')
env.reset()
res = env.step(1000)
print(res)
env.close()