# Importing required libraries
import numpy as np

# Agent here is a very simple node based neural network class that represents a single agent in the population
# input size of this class if fixed and output class is also fixed
# there are some internal neurons that are also there but there number is a input/hyperparameter
# another input to class is a hexadecimal string with batches of 8 characters (dna)
# each batch is converted to a 64 bit binary string which represents connection between all these neurons

class Agent:
    def __init__(self, hidden_neurons, dna):

        self.hidden_neurons = hidden_neurons
        self.dna = dna
        self.initialize(hidden_neurons)

    def initialize(self, hidden_neurons):
        
        # sensory neurons
        # sensory neurons can have connections to hidden and output neurons
        self.input_index = {
            "Slr": 0, # pheromone gradient left - right
            "Sfd": 1, # pheromone gradient forward,
            "Sg" : 2, # pheromone density
            "Age": 3, # age
            "Rnd": 4, # random input
            "Blr": 5, # blockage left - right
            "Osc": 6, # oscillator
            "Bfd": 7, # blockage forward
            "Plr": 8, # population gradient left - right
            "Pop": 9, # population density
            "Pfd": 10, # population gradient forward
            "LPf": 11, # population long-range forward
            "LMy": 12, # last movement y
            "LBf": 13, # blockage long-range forward
            "LMx": 14, # last movement x
            "BDy": 15, # north/south border distance
            "Gen": 16, # genetic similarity of forward neighbour
            "BDx": 17, # east/west border distance
            "Lx" : 18, # east/west world location
            "BD" : 19, # nearest border distance
            "Ly" : 20, # north/south world location
        }

        self.input_size =  len(self.input_index)
    
        # action output neurons
        self.output_index = {
            "LPD" : 0 + self.input_size, # set long-probe distance
            "Kill": 1 + self.input_size, # kill
            "OSC" : 2 + self.input_size, # set oscillator period
            "SG"  : 3 + self.input_size, # emit pheromone
            "Res" : 4 + self.input_size, # set responsiveness
            "Mfd" : 5 + self.input_size, # move forward
            "Mrn" : 6 + self.input_size, # move random
            "Mrv" : 7 + self.input_size, # move reverse
            "MRL" : 8 + self.input_size, # move left/right (+/-)
            "MX"  : 9 + self.input_size, # move east/west (+/-)
            "MY"  : 10 + self.input_size, # move north/south (+/-)
        }

        self.output_size = len(self.output_index)

        # create hidden neurons
        # hidden neurons can have connections to itself, other hidden neurons and to output neurons
        self.hidden_neurons = hidden_neurons
        self.hidden_index = {}
        for i in range(hidden_neurons):
            self.hidden_index["H_" + str(i)] = i + self.input_size + self.output_size

        self.neurons = self.input_size + self.hidden_neurons + self.output_size

        # create a map of neurons to index
        self.start_nerons = {
            "0": self.input_index,
            "1": self.hidden_index,
        }
        self.end_neurons = {
            "0": self.output_index,
            "1": self.hidden_index,
        }

        # create a weight matrix and state vector
        self.weights = self.get_weights()
        self.neuron_state = np.zeros(self.neurons)

    def get_weights(self):
        
        # dna is a list of hexadecimal string with batches of 8 characters each (32 bits)
        # each batch is converted to a 32 bit binary string which represents connection between all these neurons
        # the first 8 bits represent the connection start neuron 
        # the next 8 bits represent the connection end neuron
        # the next 16 bits represent the connection weight (0-65535), subtract 32768 to get -32768 to 32767 and divide by 32768 to get -1 to 1 range
        # in each start neuron representation the first bit represents the connection from input (0) / hidden (1) neuron
        # in each end neuron representation the first bit represents the connection to output (0) / hidden (1) neuron
        # the modulo of remaining 7 bits is used to represent the neuron index based on index length

        # we will create a input edge list from this dna and then create a weight matrix from this edge list
        weights = {}

        for batch in self.dna:

            binary_string = bin(int(batch, 16))[2:].zfill(32)

            start_neuron_index = self.start_nerons[binary_string[0]]
            start_neron_number = int(binary_string[1:8], 2) % len(start_neuron_index)
            sorted_keys = [k for k, v in sorted(self.start_neuron_index.items(), key=lambda item: item[1])]
            start_neuron = start_neuron_index[sorted_keys[start_neron_number]]

            end_neuron_index = self.end_neurons[binary_string[8]]
            end_neuron_number = int(binary_string[9:16], 2) % len(end_neuron_index)
            sorted_keys = [k for k, v in sorted(self.end_neuron_index.items(), key=lambda item: item[1])]
            end_neuron = end_neuron_index[sorted_keys[end_neuron_number]]

            weight = int(binary_string[16:], 2) - 32768
            weight = weight / 32768

            if end_neuron not in weights:
                weights[end_neuron] = {}
            
            weights[end_neuron][start_neuron] = weight

        return weights
        
    def get_outputs(self, input):

        # input is a dictionary with keys as sensory neuron names and values as neuron values
        # update the neuron state of sensory neurons
        for key, value in input.items():
            self.neuron_state[self.input_index[key]] = value

        # hidden nodes are calculated first
        for k in self.hidden_index.keys():
            
            neuron_index = self.hidden_index[k]
            weight_sum = 0

            # iterate over edge map
            for key, value in self.weights[neuron_index].items():
                weight_sum += self.neuron_state[key] * value

            # apply tanh activation function
            self.neuron_state[neuron_index] = np.tanh(weight_sum)

        # output nodes are calculated next
        output_action = {}

        for k in self.output_index.keys():
            
            neuron_index = self.output_index[k]
            weight_sum = 0

            # iterate over edge map
            for key, value in self.weights[neuron_index].items():
                weight_sum += self.neuron_state[key] * value

            # apply tanh activation function
            output_action[k] = np.tanh(weight_sum)
            self.neuron_state[neuron_index] = output_action[k]

        return output_action
    