import numpy as np

from util import accuracy
from hmm import HMM

def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	L = len(train_data)
	S = len(tags)
    ###################################################
	###### Create obs_dict ######
	obs_dict = {}
	obs_indx = 0 
	for i in range(L):
		for j in range(train_data[i].length):
			if train_data[i].words[j] not in obs_dict.keys():
				obs_dict[train_data[i].words[j]] = obs_indx 
				obs_indx += 1

	###### Create state_dict ######
	state_dict = {i:0 for i in tags}
	for i in range(S):
		state_dict[tags[i]] = i

    ###### Calculate pi ######
	pi = np.zeros(S, dtype = 'float')
	state_counts = {i:0 for i in tags} # dictionary storing counts for each tag (state)
	for i in range(len(train_data)):
		first_tag = train_data[i].tags[0] # first tag of each Line object
		state_counts[first_tag] += 1 # increase count in dictionary
	# add counts to pi array in same order as tags array
	for i in range(len(tags)):
		pi[i] = state_counts[tags[i]]
	# normalize by total number of lines to get probabilities
	pi = pi / len(train_data)
		
	###### Calculate A ######
	A = np.zeros([S, S], dtype = 'float')
	# loop through each line
	for i in range(len(train_data)):
		# loop through each element (except the last one)
		for j in range(train_data[i].length - 1):
			from_indx = state_dict[train_data[i].tags[j]]
			to_indx = state_dict[train_data[i].tags[j+1]]
			A[from_indx, to_indx] += 1
	# normalize each row of A by sum to get transition matrix 
	sum_rows = A.sum(axis = 1)
	A = A / sum_rows[:, np.newaxis]

	###### Calculate B ######
	B = np.zeros([S, len(obs_dict)])
	for i in range(len(train_data)):
		# loop through each element
		for j in range(train_data[i].length):
			s_indx = state_dict[train_data[i].tags[j]]
			o_indx = obs_dict[train_data[i].words[j]]
			B[s_indx, o_indx] += 1
	# normalize each row of B to get emissions matrix
	sum_rows = B.sum(axis = 1)
	B = B / sum_rows[:, np.newaxis]

	# initialize HMM object
	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class
    - tags: list of unique tags from pos_tags.txt (optional use)

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	S = len(model.pi)
	###################################################
	# loop through each line in data
	for i in range(len(test_data)):
		# loop through each element
		for j in range(test_data[i].length):
			# update obs_dict and emissions if there is an unseen observation
			if test_data[i].words[j] not in model.obs_dict.keys():
				o_indx = len(model.obs_dict)
				model.obs_dict[test_data[i].words[j]] = o_indx 
				new_column = np.full((S, 1), 10**-6)
				model.B = np.append(model.B, new_column, axis = 1)
		# run viterbi on each line
		Osequence = np.asarray(test_data[i].words)
		state_path = model.viterbi(Osequence)
		tagging.append(state_path)
	###################################################
	return tagging
