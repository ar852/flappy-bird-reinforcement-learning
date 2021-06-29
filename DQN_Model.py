from tensorflow.keras.layers import Conv2D, Input, Dense
from tensorflow.keras import activations
from tensorflow.keras import optimizers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import random

from constants import *



class DQN_Agent:
    """
    """

    def __init__(self, model_path=None):
        
        if model_path:
            self.q_net = keras.models.load_model(model_path)
            self.target_q_net = keras.models.load_model(model_path)
        else:
            self.q_net = self.build_dqn_model()
            self.target_q_net = self.build_dqn_model()

    def run_policy(self, state): # returns action(int)

        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net.predict(state_input)
        action = np.argmax(action_q[0], axis=0) # 0 or 1

        return action

    def run_random_policy(self): #returns action(int)
        ind = random.randint(0,99)
        return WEIGHTED_ACTIONS[ind]

    def policy(self, state, step, printing_info = False): # abstracts run_random_policy and run_policy by adding element of "uncertainty" (exploration vs exploitation)
        rand_num = random.uniform(0,1) # between 0 and 1
        explore_prob = EXPLORE_STOP + (EXPLORE_START - EXPLORE_STOP) * np.exp(-DECAY_RATE * step) # exponential decay
        if printing_info:
            print(f"Explore prob: {explore_prob}, random number: {rand_num}")
        
        if explore_prob > rand_num:
            if printing_info:
                print("Getting action through policy")
            return self.run_policy(state)
            
        else:
            if printing_info:
                print("Getting action randomly")
            return self.run_random_policy()
            

    def update_target_network(self): # update target network with other network weights. 
    # Keeping target and normal networks separate avoids "chasing" a target that is constantly moving (which makes training slow)

        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch, episode_cnt): #returns 
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net.predict(state_batch)
        target_q = np.copy(current_q)
        next_q = self.target_q_net.predict(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)

        for i in range(state_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += GAMMA * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val

        #csv_logger = tf.keras.callbacks.CSVLogger("csv log/csv logger.csv", separator=',', append=False)
        # ckpt_logger = tf.keras.callbacks.ModelCheckpoint( \
        #     filepath=f'ckpts/checkpoint{episode_cnt}', monitor='loss', verbose=1, save_best_only=True, \
        #     save_weights_only=False, mode='auto', save_freq='epoch',options=None)
        history_logger = tf.keras.callbacks.History()

        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose=0, \
            callbacks=[history_logger])
       
        loss = training_history.history['loss']
        return loss

    def save_model(self, name):
        self.q_net.save(f"ckpts/checkpoint {name}")

    @staticmethod
    def build_dqn_model():

        frames_input = Input(shape=STATE_SIZE, name="frames") 

        normalized = keras.layers.Lambda(lambda x: x/255.0)(frames_input) # no9rmalizatin

        # Conv 1: 16 (8,8) filters  with (4,4) stride --> Output: (None, 15, 10, 16)
        conv_1 = Conv2D(16, 8, strides=(4,4), activation='relu', name='conv_1') (normalized)

        # Conv 2: 32 (8,8) filters with (4,4) strides --> Output: (None, 17, 9, 32)
        conv_2 = Conv2D(32, 4, strides=(2,2), activation='relu', name='conv_2') (conv_1)

        # Flatten --> Output: (None, 4896)
        flattened = keras.layers.Flatten(name="flattened") (conv_2)

        # Dense 200 Nodes --> Output (None, 200)
        dense_hidden = keras.layers.Dense(128, activation='relu', name='dense_hidden') (flattened)

        # Output (None, 2)
        output = keras.layers.Dense(ACTION_SIZE, activation='linear', name='output', dtype=tf.float32) (dense_hidden)

        # Optimizer
        opt = optimizers.Adam(lr=LEARNING_RATE)

        model = keras.models.Model(inputs=frames_input, outputs=output, name='DQN_Model')

        model.compile(opt, loss='mse')
        
        #print(model.summary())

        return model


def testing_policy():

    agent = DQN_Agent()
    state = np.ones(STATE_SIZE)
    print("\n\n\n\n")
    print("Runing through qnet")
    print(agent.q_net.predict(np.expand_dims(state, 0)))
    print("Running through policy")
    print(agent.run_policy(state))
    print("\n")
    print("Running through policy with uncertainty")
    for i in range(10):
        print(agent.policy(state, True)) 

if __name__ == '__main__':
    DQN_Agent() # to print the model summary (which you should comment out after)
