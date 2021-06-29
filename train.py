import flappy_bird_gym
import time as t
from collections import deque
from PIL import Image
import numpy as np
import time
import pickle
from datetime import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


from constants import *
from replay_buffer import Replay_Buffer
from DQN_Model import DQN_Agent

def preprocess(img_arr):

    img_arr = img_arr.astype(np.uint8) # 0 to 255 like rgb values
    # tranpose and crop --> (288,512,3) to (407, 288, 3) 
    img_arr = img_arr.transpose((1,0,2))[:-105, :] 
    #print(f"Shape after transpose crop: {img_arr.shape}")
    # Cvt to gray and resize
    img = Image.fromarray(img_arr).convert("L")
    img = img.resize(tuple(reversed(PROCESSED_IMG_SIZE))) 
    #print(f"Resized image shape: {img.size}")
    #print(np.array(img).shape)
    return np.array(img) # np.array

def update_stack(dq, frame, first_frame = False):

    if first_frame:

        dq = deque( [np.zeros(PROCESSED_IMG_SIZE, dtype=np.int) for i in range(STACK_SIZE)] , maxlen=STACK_SIZE)
        for i in range(STACK_SIZE):
            dq.append(frame)
        stacked_array = np.dstack(dq) # (150, 84, 4)

    else:

        dq.append(frame)
        stacked_array = np.dstack(dq) # (150, 84, 4)

    return dq, stacked_array

def get_stack(dq, img_arr, first_frame): # abstracts functions process and update_stack

    preprocessed_arr = preprocess(img_arr)

    if first_frame:
        dq, stacked_array = update_stack(dq, preprocessed_arr, first_frame = True)
    else: 
        dq, stacked_array = update_stack(dq, preprocessed_arr, first_frame = False)

    return dq, stacked_array


def collect_experiences(env, agent, buff, step, rate = 30, render = False, print_score=False):

    dq = deque([], maxlen=STACK_SIZE)
    state = env.reset()
    dq, state = get_stack(dq, state, True)
    done = False

    curr_score = 0
    last_score = 0
    while not done:
        action = agent.policy(state, step)

        if render:
            env.render()
            framerate(rate)

        next_state, reward, done, info = env.step(action) # standard is 1
        dq, next_state = get_stack(dq, next_state, False)
        curr_score = info['score']
        reward = 1
        if curr_score > last_score:
            reward = 50

        if done:
            reward = -1000

        buff.store(state, next_state, reward, action, done)
        state = next_state
        last_score = curr_score

        if print_score:
            print(f"[INFO] {info['score']}, {reward}, {action}, {done} ")

def framerate(rate):
    time.sleep(1/rate)

def testing_agent(env, agent, step): # Tests agent's score
    dq = deque([], maxlen=STACK_SIZE)
    total_score = 0.0
    eps = TESTING_EPISODES

    for i in range(eps):
        state = env.reset()
        dq, state = get_stack(dq, state, True)
        done = False
        episode_score = 0.0

        while not done:
            action = agent.policy(state, step)
            next_state, _, done, info = env.step(action)
            dq, next_state = get_stack(dq, next_state, False)
            episode_score = info['score']
            state = next_state

        total_score += episode_score

    average_score = total_score / eps
    return average_score




def train_model(max_eps = 100000): # big boy function

    agent = DQN_Agent()
    buff = Replay_Buffer(size=1000000) # 1M
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    precollect_num = 200
    step = 0
    metrics = []

    for i in range(precollect_num): # get some data in so we don't sample from only like 2 samples when we train
        collect_experiences(env, agent, buff, step)
        print(f"\tPrecollecting experience {i}/{precollect_num}")

    print("\n")

    start_training = dt.now().strftime("%H_%M_%S  %m-%d-%Y")
    save_time = time.time()

    for episode_cnt in range(max_eps):

        # if (time.time() - save_time >= 5):
        #     plt.close('all') # close plot

        collect_experiences(env, agent, buff, step, print_score=False)
        gameplay_experience_batch = buff.sample()
        loss = agent.train(gameplay_experience_batch, episode_cnt)
        avg_score = testing_agent(env, agent, step)
        print('\tEpisode {0}/{1}, explore prob {2}, avg score is {3}, and ' \
              'loss is {4}'.format(episode_cnt, max_eps, EXPLORE_STOP + (EXPLORE_START - EXPLORE_STOP) * np.exp(-DECAY_RATE * step), # exponential decay, \
                                   avg_score, loss[0]))

        metrics.append((loss[0], avg_score))

        if episode_cnt % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
        if episode_cnt % MODEL_SAVE_FREQUENCY == 0:
            now = dt.now()
            ckpt_str = now.strftime("%H_%M_%S  %m-%d-%Y")
            metrics_str = now.strftime("%m-%d-%Y")
            agent.save_model(ckpt_str)
            with open(f"metrics {metrics_str}", mode='wb') as file:
                pickle.dump(metrics, file)
            #make_plot(f"metrics {now}")
            save_time = time.time()

        step += 1

    end_training = dt.now().strftime("%m-%d-%Y")
    agent.save_model(f"Finished Training {now}")
    env.close()
    print("Started training at {start_training}\nEnded training at {end_training}")

# ============= TESTING FUNCTIONS =================

def testing_get_stack():
    dq = deque([], maxlen=STACK_SIZE)
    img_arr = (np.random.rand(288, 512, 3) * 255).astype(np.uint8)
    print(f"Input image shape: {img_arr.shape}")
    dq, stacked_array = get_stack(dq, img_arr, first_frame=True)
    print(f"Stacked array shape: {stacked_array.shape}")

def testing_collect_experiences():
    agent = DQN_Agent()
    buff = Replay_Buffer(size=100000) # 10M
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    for i in range(10):
        collect_experiences(env, agent, buff, render=False, print_score=True) #Rendering is glitchy
        env.reset()
    return buff

# buff = testing_collect_experiences()
# print("Length of buffer: %d" % len(buff.gameplay_experiences))

if __name__ == '__main__':
    train_model()