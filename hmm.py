from __future__ import print_function
import numpy as np


class HMM:
    
    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
            - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
            - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
            - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
            - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
            - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
            """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
    
    def forward(self, Osequence):
        """
            Inputs:
            - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
            - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
            - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
            - Osequence: (1*L) A numpy array of observation sequence with length L
            
            Returns:
            - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
            """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Initialize alpha (t=1)
        for i in range(S):
            alpha[i, 0] = self.pi[i] * self.B[i, self.obs_dict[Osequence[0]]]
        
        # Iterate forward (t = 2 to T)
        for t in range(1, L):
            for i in range(S):
                local_sum = 0
                # loop through all possible previous states
                for j in range(S):
                    local_sum += self.A[j, i] * alpha[j,t-1]
                alpha[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * local_sum
        ###################################################
        return alpha
    
    def backward(self, Osequence):
        """
            Inputs:
            - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
            - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
            - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
            - Osequence: (1*L) A numpy array of observation sequence with length L
            
            Returns:
            - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
            """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Initialize (t = T)
        for i in range(S):
            beta[i, L-1] = 1
        
        # Iterate backwards (t = (T-1) to 1)
        for t in range(L-2, -1, -1):
            for i in range(S):
                # loop through all states (s')
                for j in range(S):
                    beta[i, t] += self.A[i, j] * self.B[j, self.obs_dict[Osequence[t+1]]] * beta[j, t+1]
        ###################################################
        return beta
    
    def sequence_prob(self, Osequence):
        """
            Inputs:
            - Osequence: (1*L) A numpy array of observation sequence with length L
            
            Returns:
            - prob: A float number of P(x_1:x_T | λ)
            """
        prob = 0
        ###################################################
        # compute alpha
        alpha_matrix = self.forward(Osequence)
        alpha_T = alpha_matrix[:, len(Osequence)-1]
        
        # compute probability of observed sequence
        prob = np.sum(alpha_T)
        ###################################################
        return prob
    
    def posterior_prob(self, Osequence):
        """
            Inputs:
            - Osequence: (1*L) A numpy array of observation sequence with length L
            
            Returns:
            - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
            """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # compute alpha, beta, probability of observed sequence
        alpha_matrix = self.forward(Osequence)
        beta_matrix = self.backward(Osequence)
        p_seq = self.sequence_prob(Osequence)
        
        # compute posterior
        # loop over time steps (t)
        for t in range(L):
            # loop over previous states (s')
            for i in range(S):
                prob[i, t] = (alpha_matrix[i, t] * beta_matrix[i, t]) / p_seq
        ###################################################
        return prob
    
    def likelihood_prob(self, Osequence):
        """
            Inputs:
            - Osequence: (1*L) A numpy array of observation sequence with length L
            
            Returns:
            - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
            """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # compute alpha, beta, probability of observed sequence
        alpha_matrix = self.forward(Osequence)
        beta_matrix = self.backward(Osequence)
        p_seq = self.sequence_prob(Osequence)
        
        # loop over time steps (t)
        for t in range(L-1):
            # loop over previous states (s')
            for i in range(S):
                # loop over current states (s)
                for j in range(S):
                    # compute numerator of likelihood
                    prob[j, i, t] = alpha_matrix[j, t] * self.A[j, i] * self.B[i, self.obs_dict[Osequence[t+1]]] * beta_matrix[i, t+1]
        
        # calculate final expression of likelihood (ksi)
        prob = prob/p_seq
        ###################################################
        return prob
    
    def viterbi(self, Osequence):
        """
            Inputs:
            - Osequence: (1*L) A numpy array of observation sequence with length L
            
            Returns:
            - path: A List of the most likely hidden state path k* (return state instead of idx)
            """
        path = []
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros(shape = (S, L), dtype = 'float')
        delta_table = np.empty(shape = (S, L-1), dtype='int')
        ###################################################
        # Initialize delta for each state (this is an array with length S)
        # self.B[self.obs_dict[Osequence[0]]] = column which corresponds to the first state of Osequence
        delta[:,0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        
        # loop through time steps (t)
        for t in range(1, L):
            # loop through states
            for i in range(S):
                # find and store max
                # loop through previous states (s')
                local_max = -1
                for j in range(S):
                    current_max = self.A[j, i] * delta[j, t-1]
                    if current_max > local_max:
                        local_max = current_max
                        delta_table[i, t-1] = j
                delta[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * local_max
        
        # Backtracking to get list of states
        path.append(np.argmax(delta[:, -1])) # initialize with last state at t = T
        for t in range(L-1, 0, -1):
            path.append(delta_table[path[len(path)-1] ,t-1])
        path.reverse()
        
        for i in range(0, len(path)):
            for key, value in self.state_dict.items():
                if value == path[i]:
                    path[i] = key
        ###################################################
        return path
