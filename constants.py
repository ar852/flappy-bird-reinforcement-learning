ACTION_SIZE = 2
STACK_SIZE = 4
TESTING_EPISODES = 4 # run after each epoch of training

PROCESSED_IMG_SIZE = (81, 56) # 5 times reduction, was originally 3 but that took even longer to train
STATE_SIZE = (81, 56, 4)

EXPLORE_START = 0.5 
EXPLORE_STOP = 0.02
DECAY_RATE = 0.001 
WEIGHTED_ACTIONS = [1] * 40 + [0] * 60 # dont flap is more common than flap
LEARNING_RATE = 0.0005 # for adam optimizer (should this be smaller?)
GAMMA = 0.95 
TARGET_UPDATE_FREQUENCY = 100
MODEL_SAVE_FREQUENCY = 1000

BATCH_SIZE = 128 # Based off of what other people used from our research

def make_plot(path):
	import pickle
	import pandas as pd
	from matplotlib import pyplot as plt

	with open(path, "rb") as file:
		tup = pickle.load(file)
	
	df = pd.DataFrame(tup, columns=["Loss", "Score"])
	fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12,8))
	axes[0].plot(df.Loss)
	axes[1].plot(df.Score)
	plt.show()

make_plot("metrics 05-06-2021")