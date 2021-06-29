import flappy_bird_gym
from collections import deque

from DQN_Model import DQN_Agent
from train import framerate, get_stack
from constants import *

model_folder_name = "checkpoint 13_12_59  05-06-2021" # change based on model
numGames = 200
scores = []

def run_multiple_games():
	agent = DQN_Agent(f"ckpts/{model_folder_name}")
	env = flappy_bird_gym.make("FlappyBird-rgb-v0")

	for i in range(numGames):
		print(f"Running Game {i}")
		dq = deque([], maxlen=STACK_SIZE)
		state = env.reset()
		dq, state = get_stack(dq, state, True)
		done = False
		while not done:
			#env.render()
			action = agent.policy(state)
			next_state, reward, done, info = env.step(action)
			dq, next_state = get_stack(dq, next_state, False)
			state = next_state
			#framerate(60)
		print(info['score'])
		scores.append(info['score'])

	print(sorted(scores))
	print(f"Avg score: {sum(scores)/len(scores)}")

def run_one_game():
	agent = DQN_Agent(f"ckpts/{model_folder_name}")
	env = flappy_bird_gym.make("FlappyBird-rgb-v0")
	dq = deque([], maxlen=STACK_SIZE)
	state = env.reset()
	dq, state = get_stack(dq, state, True)
	done = False
	while not done:
		env.render()
		action = agent.policy(state)
		next_state, reward, done, info = env.step(action)
		dq, next_state = get_stack(dq, next_state, False)
		state = next_state
		framerate(60)


if __name__=='__main__':
	run_one_game()