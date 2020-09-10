import gym
import random
import torch
import numpy as np
from dqn_agent import Agent
import imageio
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw 

def DQN_gif(file_name):
    env = gym.make('LunarLander-v2')
    env.seed(0)

    agent = Agent(state_size=8, action_size=4, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth', map_location=lambda storage, loc: storage))

    images = []
    state = env.reset()
    img = env.render(mode='rgb_array')
    for j in range(200):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        frame = env.render(mode='rgb_array')
        
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        text = 'Step = {}\nReward = {}'.format(j+1, reward)
        draw.text((20, 20),text,(255,255,255))

        images.append(np.asarray(pil_img))
        
        if done:
            break 
    imageio.mimsave(file_name, images)
    
if __name__ == '__main__':
    DQN_gif(file_name='lunar_lander.gif')
    