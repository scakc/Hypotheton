# Import required libraries
import numpy as np

# An env class for evolution experiment 
# contains a world of 128 x 128 cells
# world will have a population of agents from the Agent class in agent.py
# world can have cells marked as safe zone and danger zone 
# at end of each generation, agents will be evaluated based on their performance in the world
# if in danger zone, agent will be removed
# if in safe zone, agent will be able to reproduce
# reproduction will be asexual with mutation in dna of child agent 
# dna of agent is a hex string, mutation will be a random change in this string
# each cell can have a pheromone level, agent, danger signal, blockage signal (restricted movement like border)

class SimpleEvolutionEnv:

    def __init__(self, board_size = [128, 128], max_steps_per_generation = 300):

        self.world_size = board_size
        self.max_steps_per_generation = max_steps_per_generation
        self.reset()
    
    def reset(self):
        
        self.step = 0
        self.world = np.zeros(self.world_size)
        self.pheromone = np.zeros(self.world_size)
        self.danger = np.zeros(self.world_size)
        self.blockage = np.zeros(self.world_size)
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def step(self):
        self.step += 1
        for agent in self.agents:
            agent.step(self)
        if self.step > self.max_steps_per_generation:
            self.evaluate()
            self.reset()

    def evaluate(self):
        pass
    
    def render(self):
        pass

    

