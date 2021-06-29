

import numpy as np
import time as t
import random as rand
from PIL import Image
# import tensorflow as tf
# from tensorflow.keras import preprocessing
import time as t
import flappy_bird_gym as f
from collections import deque
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONSTANTS =========================

WEIGHTED_ACTIONS = [1] * 10 + [0] * 100 # to not hit the damn ceiling every time
STACK_SIZE = 4
PROCESSED_IMG_SIZE = (136, 96)

# ===================== FUNCTIONS =========================

def preprocess(img_arr):

    img_arr = img_arr.astype(np.uint8) # 0 to 255 like rgb values
    # tranpose --> (288,512,3) to (512, 288, 3) and crop useless bottom pixels
    img_arr = img_arr.transpose((1,0,2))[:-105, :] 
    print(f"Shape after transpose crop: {img_arr.shape}")
    # Cvt to gray and resize
    img = Image.fromarray(img_arr).convert("L")
    img = img.resize(PROCESSED_IMG_SIZE)
    print(f"Resized image shape: {img_arr.shape}")
    #print(np.array(img).shape)
    return img # np.array

def create_environment(rgb=True):

    if rgb:
        env = f.make("FlappyBird-rgb-v0")
    else:
        env = f.make("FlappyBird-v0")
    state = env.reset()
    print(state.shape)
    possible_actions = [0,1]  # down, up (respectively)
    return env, possible_actions

# Deepmind paper adds to stack every 4 frames 
def update_stack(dq, frame, first_frame = False):

    if first_frame:

        dq = deque( [np.zeros(PROCESSED_IMG_SIZE, dtype=np.int) for i in range(STACK_SIZE)] , maxlen=STACK_SIZE)
        dq.append(frame)
        dq.append(frame)
        dq.append(frame)
        dq.append(frame)
        stacked_array = np.dstack(dq) # (150, 84, 4)

    else:

        dq.append(frame)
        stacked_array = np.dstack(dq) # (150, 84, 4)

    return dq, stacked_array

def framerate(frames):

    t.sleep(1/(frames+3))


# ===================== INITIALIZATION =========================

def run_one_game():

    env = flappy_bird_gym.make("FlappyBird-rgb-v0")

    while True:
        env.render()
        action = rand.choice(WEIGHTED_ACTIONS)
        obs, reward, done, info = env.step(action)
        print(obs.shape, reward, info, sep="\n", end="\n\n")
        if done:
            break
        framerate(60)

    env.close()
    preprocess(obs).show()

def make_plot(path):
    with open(path, "rb") as file:
        tup = pickle.load(file)
    
    df = pd.DataFrame(tup, columns=["Loss", "Score"])
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12,8))
    axes[0].plot(df.Loss)
    axes[1].plot(df.Score)
    plt.show()


make_plot()


"""

import gym
env = gym.make('MsPacman-v0')
env.reset()
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done==True:
        break
env.close()

"""