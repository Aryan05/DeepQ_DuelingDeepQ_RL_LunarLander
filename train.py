from dqn_agent import Agent


env = gym.make('LunarLander-v2')
env.seed(0)

agent = Agent(state_size=8, action_size=4, seed=0)