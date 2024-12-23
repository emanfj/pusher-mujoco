import random
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from HERCompatiblePusherWrapper import HERCompatiblePusherWrapper  


def save_video(model):
    env = gym.make("Pusher-v5", render_mode="rgb_array")  #use 'rgb_array' to save video frames
    # uncomment below line while using SAC+HER
    # env = HERCompatiblePusherWrapper(env)

    #wrappper for recording video
    env = gym.wrappers.RecordVideo(env, './videos', name_prefix="ppo_pusher")

    #Reset the environment to start recording
    observation, info = env.reset()
    done = False

    i = 0
    while not done:
        action, _states = model.predict(observation, deterministic=False)
        observation, rewards, terminated, truncated, info = env.step(action)
        
        #done = terminated or truncated
        i += 1
        if i > 1000: #terminate after 1000 steps
            break

    env.close()

def show_run(model):
    
    #does not auto terminate user has to terminate
    env = gym.make("Pusher-v5", render_mode="human") 
    observation, info = env.reset()
    done = False

    while not done:
    
        action, _states = model.predict(observation, deterministic=False)
        observation, rewards, terminated, truncated, info = env.step(action)
        
        env.render()
        
        #done = terminated or truncated

    env.close()

model = PPO.load("ppo_pusher")

#to run the model
show_run(model)

#to save video of the model
save_video(model)