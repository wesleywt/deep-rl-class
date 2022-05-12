import gym

# First, we create our environment called LunarLander-v2
env = gym.make("LunarLander-v2")

# Then we reset this environment
observation = env.reset()

for _ in range(20):
    # Take a random action
    action = env.action_space.sample()
    print("Action taken:", action)

    # Do this action in the environment and get
    # next_state, reward, done and info
    observation, reward, done, info = env.step(action)

    # If the game is done (in our case we land, crashed or timeout)
    if done:
        # Reset the environment
        print("Environment is reset")
        observation = env.reset()