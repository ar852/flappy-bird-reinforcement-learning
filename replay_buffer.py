import random 
from collections import deque
import numpy as np
from constants import *

class Replay_Buffer:

	def __init__(self, size):
		self.size = size
		self.gameplay_experiences = deque(maxlen=self.size)

	def store(self, state, next_state, reward, action, done):

		self.gameplay_experiences.append((state, next_state, reward, action, done))

	def sample(self):
		batch_size = min(BATCH_SIZE, len(self.gameplay_experiences)) #whichever is bigger
		sampled_gameplay_batch = random.sample(self.gameplay_experiences, batch_size)

		state_batch = []
		next_state_batch = []
		action_batch = []
		reward_batch = []
		done_batch = []

		for gameplay_experience in sampled_gameplay_batch:
			state_batch.append(gameplay_experience[0])
			next_state_batch.append(gameplay_experience[1])
			reward_batch.append(gameplay_experience[2])
			action_batch.append(gameplay_experience[3])
			done_batch.append(gameplay_experience[4])

		return np.array(state_batch), np.array(next_state_batch), np.array( \
			action_batch), np.array(reward_batch), np.array(done_batch)

# ============== TESTING ==================

def buffer_testing():

	buff = Replay_Buffer(size=1000)
	for i in range(20):
		buff.store(*np.random.rand(5))
	print(buff.sample())

