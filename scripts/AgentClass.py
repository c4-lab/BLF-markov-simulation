"""Each agent has coherence matrix and neighbors which are agents as well"""
import utilities
import numpy as np
from scipy.stats import logistic

class Agent:
    
    def __init__(self, name='', tow=-1, knowledge_dim = 3, initial_state=None):
        self.neighbors = {} # neighbors will also be dictionary of agents
        self.name = name
        self.tow = tow # agent's threshold defined in initialization
        
        if initial_state: # if provided with initial knowledge state
            if isinstance(initial_state, list):
                self.knowledge_state = initial_state
            else:
                raise ValueError('list expected')
        self.dissonance_lst = None  
        self.next_state = None
       
    def add_neighbors(self, neighbor_agent):
        """add neigbhor agent to the dictionary of neighbors
        which has neighbor agent name for key and agent itself as value"""
        self.neighbors[neighbor_agent.name] = neighbor_agent
        neighbor_agent.neighbors[self.name] = self # neighbors also get new neighbors
    
    def get_neighbors(self):
        """return all neighbors for the agent"""
        return self.neighbors
    
    def get_neighbors_name(self):
        return [n.name for k, n in self.neighbors.items()]
       
    def remove_neighbors(self, name):
        """remove neighbor for the agent with name"""
        if name in self.neighbors:
            del self.neighbors[name]
            print('removed neighbor')
        else:
            print('neighbor not found')
            
    
    def update_knowledge(self, alpha, coherence_matrix):
        """Takes alpha and coherence matrix"""
        # first convert state binary to int to get the row in coherence matrix
        row_ptr = utilities.bool2int(self.knowledge_state)
        # get the corresponding probabilites from the matrix
        prob_flip_bit = coherence_matrix[row_ptr]
        dissonance_list = []
        next_state = []
        for index, curr_bit_state in enumerate(self.knowledge_state):
            # now look for neighbors who agree in this bit value
            # TODO THis needs to be dissimilarity, not similarity
            neigh_agreement_count = self.count_neigbors_adapted_knowledge(index)
            
            # compute d as (# of neighbors agree on bit/# of neighbors)
            if len(self.neighbors) > 0:
                d = neigh_agreement_count/len(self.neighbors)
            else:
                d = 0
            dissonance = logistic.cdf(d-self.tow)
            dissonance_list.append(dissonance)
        
        for i in range(len(dissonance_list)):
            flip_probability = alpha*dissonance_list[i] + (1-alpha)*prob_flip_bit[i] 
            rand_num = np.random.rand()
            if flip_probability > rand_num:
                f  = 1^self.knowledge_state[i]
                next_state.append(f)
            else:
                next_state.append(self.knowledge_state[i])
            
        self.next_state = next_state
        self.dissonance_lst = dissonance_list
    
    def count_neigbors_adapted_knowledge(self, kbit):
        count = 0
        for name, agent in self.neighbors.items():
            if agent.knowledge_state[kbit] == self.knowledge_state[kbit]:
                count += 1
                
        return count
    
    
    def __str__(self):
        print('-'*30)
        s = 'agent name : ' + self.name
        s += '\nknowledge_state: [' + ', '.join(map(str, self.knowledge_state)) + ']'
        for n,v in self.neighbors.items():
            s += '\n'
            s += self.name + ' <-> ' + v.name
            
        return s