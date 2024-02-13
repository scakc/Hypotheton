# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
import multiprocessing
import pandas as pd
import time
import concurrent.futures

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

    def __init__(self, board_size = [128, 128], max_steps_per_generation = 300, population = 1000, number_hidden_neurons = 1, env_type = "simple", gene_length = 4):

        self.world_size = board_size
        self.max_steps_per_generation = max_steps_per_generation
        self.max_population = population
        self.number_hidden_neurons = number_hidden_neurons
        self.env_type = env_type
        self.generation_count = 0
        self.population = 0
        self.gene_length = gene_length
        self.survival_rate = 0
        self.agent_indexes = np.array([-1]*self.max_population)

        self.dna_bank = [[str(format(np.random.randint(0, 16**8, dtype=np.int64) , '08x')) for _ in range(self.gene_length)] for _ in range(self.max_population)]

        # 9 movement directions
        self.movement_names = {
            "stay": 0, "west": 1, "north-west": 2, "north": 3, "north-east": 4, "east": 5, "south-east": 6, "south": 7, "south-west": 8
        }
        self.direction_index = {
            # each is corresponding movement in x and y as [x, y]
            0: np.array([0, 0]), 
            1: np.array([-1, 0]), 
            2: np.array([-1, 1]), 
            3: np.array([0, 1]), 
            4: np.array([1, 1]), 
            5: np.array([1, 0]), 
            6: np.array([1, -1]), 
            7: np.array([0, -1]), 
            8: np.array([-1, -1])
        }

        self.input_keys = ["Slr", "Sfd", "Sg", "Age", "Rnd", "Blr", "Osc", "Bfd", "Plr", "Pop", "Pfd", "LPf", "LMy", "LBf", "LMx", "BDy", "Gen", "BDx", "Lx", "BD", "Ly"]
        self.output_keys = ["LPD", "Kill", "OSC", "SG", "Res", "Mfd", "Mrn", "Mrv", "MRL", "MX", "MY"]

        self.pheromone_decay_rate = 0.8
        # self.random_states = [{key: np.random.rand() for key in self.input_keys} for _ in range(self.max_population)]
        self.reset()
    
    def reset(self, keep_old_agents = True):

        if keep_old_agents:
            self.generation_count += 1
        
        self.current_step = 0
        self.killings = 0
        self.world = np.zeros(self.world_size)
        self.new_world = np.zeros(self.world_size)
        self.pheromone = np.zeros(self.world_size)
        self.danger = np.zeros(self.world_size)
        self.blockage = np.zeros(self.world_size)
        
        # Add danger zone and blockage
        if self.env_type == "simple":
            self.danger[:45, -80:] = 1
            self.danger[-45:, :80] = 1
            self.danger[40:-40, -80:80] = 1
            self.blockage[:90, 40:45] = 1
            self.blockage[-90:, -45:-40] = 1
            self.danger[:, 20:65] = 1
            self.danger[:, -65:-20] = 1
            # self.danger[50:-50, 55:-55] = 0

        if keep_old_agents:
            # print("Keeping old agents")
            dna_bank = self.dna_bank
        else:
            dna_bank = [[str(format(np.random.randint(0, 16**8, dtype=np.int64) , '08x')) for _ in range(self.gene_length)] for _ in range(self.max_population)]

        # random placement of agents where there is no blockage
        self.agents = [] 
        self.population = 0
        pop_counter = 0
        self.position_agents = {} 
        while pop_counter < self.max_population:
            x = np.random.randint(self.world_size[0])
            y = np.random.randint(self.world_size[1])

            if self.blockage[x, y] == 0 and self.world[x, y] == 0:
                # Create agents with dna
                dna_agent = dna_bank[np.random.choice(np.arange(0,len(dna_bank)))]
                # print("Adding agent with dna", dna_agent)
                self.add_agent(Agent(self.number_hidden_neurons, dna_agent), x, y)
                pop_counter += 1
                
        return self.get_state()

    def destroy_unsafe_agents(self):
        # remove agents in danger zone
        for k, agent in enumerate(self.agents):
            if self.danger[agent.x, agent.y] == 1:
                self.remove_agent(agent)

        self.survival_rate = self.population / self.max_population

        # save dnas
        self.dna_bank = [agent.get_dna() for agent in self.agents]

    def get_state(self):

        # return self.get_batch_state()

        # pheromone density sum of surrounding 8 cells, requires convolution
        padded_size = (self.world_size[0] + 2, self.world_size[1] + 2)
        pheromone_density = np.zeros(padded_size)
        population_density = np.zeros(padded_size)
        blockage_density = np.zeros(padded_size)
        padded_pheromone = np.pad(self.pheromone, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        padded_population = np.pad(self.world, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        padded_blockage = np.pad(self.blockage, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        for i in range(8):
            pheromone_density += np.roll(padded_pheromone, self.direction_index[i], axis=(0, 1))
            population_density += np.roll(padded_population, self.direction_index[i], axis=(0, 1))
            blockage_density += np.roll(padded_blockage, self.direction_index[i], axis=(0, 1))

        self.pheromone_density = pheromone_density[1:-1, 1:-1]
        self.population_density = population_density[1:-1, 1:-1]
        self.blockage_density = blockage_density[1:-1, 1:-1]

        # return self.get_state_multi()
        agent_states = [self.get_features(agent) for agent in self.agents]
        return agent_states

    def get_cell_value(self, property, x, y, def_value = 0):

        if x < 0 or x >= self.world_size[0] or y < 0 or y >= self.world_size[1]:
            return def_value
        else:
            return property[x, y]
        
    def set_cell_value(self, property, x, y, value):
        if x >= 0 and x < self.world_size[0] and y >= 0 and y < self.world_size[1]:
            property[x, y] = value
        
    def get_box_value(self, property, x, y, box_size = [1,1,1,1]):
        # get sum of box_size x box_size cells around x, y
        x_start = x - box_size[0] if x - box_size[0] >= 0 else 0
        x_end = x + box_size[1] if x + box_size[1] < self.world_size[0] else self.world_size[0]
        y_start = y - box_size[2] if y - box_size[2] >= 0 else 0
        y_end = y + box_size[3] if y + box_size[3] < self.world_size[1] else self.world_size[1]

        return property[x_start:x_end, y_start:y_end]

    def add_pheromone_value(self, x, y, value, box_size = [1,1,1,1]):
        # set sum of box_size x box_size cells around x, y

        x_start = x - box_size[0] if x - box_size[0] >= 0 else 0
        x_end = x + box_size[1] if x + box_size[1] < self.world_size[0] else self.world_size[0]
        y_start = y - box_size[2] if y - box_size[2] >= 0 else 0
        y_end = y + box_size[3] if y + box_size[3] < self.world_size[1] else self.world_size[1]

        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                self.pheromone[i, j] += value

    def get_probe_value(self, property, x, y, direction, probe_distance = 5):

        probe_values = []
        for i in range(1, int(probe_distance) + 1):
            probe_values.append(self.get_cell_value(property, x + i * direction[0], y + i * direction[1]))

        return probe_values

    def get_features(self, agent):
        
        # input_keys = ["Slr", "Sfd", "Sg", "Age", "Rnd", "Blr", "Osc", "Bfd", "Plr", "Pop", "Pfd", "LPf", "LMy", "LBf", "LMx", "BDy", "Gen", "BDx", "Lx", "BD", "Ly"]
        # sample_input = {key: np.random.rand() for key in input_keys}
        # return sample_input
        
        x, y = agent.x, agent.y
        move_direction = self.direction_index[agent.direction]

        env_features = {}

        move_x = move_direction[0]
        move_y = move_direction[1]

        # left moves
        left_x, left_y = x - move_y, y + move_x
        left_x = (left_x if left_x >= 0 else 0) if left_x < self.world_size[0] else self.world_size[0] - 1
        left_y = (left_y if left_y >= 0 else 0) if left_y < self.world_size[1] else self.world_size[1] - 1
        
        # right moves
        right_x, right_y = x + move_y, y - move_x
        right_x = (right_x if right_x >= 0 else 0) if right_x < self.world_size[0] else self.world_size[0] - 1
        right_y = (right_y if right_y >= 0 else 0) if right_y < self.world_size[1] else self.world_size[1] - 1

        # forward moves
        forward_x = x + move_x
        forward_x = (forward_x if forward_x >= 0 else 0) if forward_x < self.world_size[0] else self.world_size[0] - 1
        forward_y = y + move_y
        forward_y = (forward_y if forward_y >= 0 else 0) if forward_y < self.world_size[1] else self.world_size[1] - 1
        forward_x_long = x + int(agent.long_probe_distance) * move_x
        forward_x_long = (forward_x_long if forward_x_long >= 0 else 0) if forward_x_long < self.world_size[0] else self.world_size[0] - 1
        forward_y_long = y + int(agent.long_probe_distance) * move_y
        forward_y_long = (forward_y_long if forward_y_long >= 0 else 0) if forward_y_long < self.world_size[1] else self.world_size[1] - 1

        # pheromone gradient left - right
        left_peromone = self.pheromone[left_x, left_y]
        right_peromone = self.pheromone[right_x, right_y]
        env_features["Slr"] = left_peromone - right_peromone

        # pheromone gradient forward
        forward_peromone = self.pheromone[forward_x, forward_y]
        current_peromone = self.pheromone[x, y]
        env_features["Sfd"] = current_peromone - forward_peromone

        # pheromone density
        env_features["Sg"] = self.pheromone_density[x, y]

        # age
        env_features["Age"] = agent.age

        # random input
        env_features["Rnd"] = np.random.rand()

        # blockage left - right
        left_blockage = self.blockage[left_x, left_y]
        right_blockage = self.blockage[right_x, right_y]
        env_features["Blr"] = left_blockage - right_blockage # does not work if both are blocked (assuming will never happen)
        
        # oscillator
        env_features["Osc"] = agent.oscillator

        # blockage forward
        env_features["Bfd"] = self.blockage[forward_x, forward_y]

        # population gradient left - right
        left_population = self.world[left_x, left_y]
        right_population = self.world[right_x, right_y]
        env_features["Plr"] = left_population - right_population

        # population density sum of surrounding 8 cells
        env_features["Pop"] = self.population_density[x, y]

        # population gradient forward
        forward_population = self.world[forward_x, forward_y]
        current_population = self.world[x, y]
        env_features["Pfd"] = forward_population - current_population

        # population long-range forward
        env_features["LPf"] = self.population_density[forward_x_long, forward_y_long]

        # last movement y
        env_features["LMy"] = y - move_y

        # blockage long-range forward
        env_features["LBf"] = self.blockage_density[forward_x_long, forward_y_long]

        # last movement x
        env_features["LMx"] = x - move_x

        # north/south border distance
        env_features["BDy"] = min(y, self.world_size[1] - y)

        # genetic similarity of forward neighbour
        env_features["Gen"] = 0

        # east/west border distance
        env_features["BDx"] = min(x, self.world_size[0] - x)

        # east/west world location
        env_features["Lx"] = x

        # nearest border distance
        env_features["BD"] = min(env_features["BDx"], env_features["BDy"])

        # north/south world location
        env_features["Ly"] = y

        return env_features

    def add_agent(self, agent, x, y):
        
        if self.world[x, y] == 1:
            print("Agent already present at", x, y)
            assert False

        agent.set_position(x, y)
        agent.mutate_dna(0.01)
        self.position_agents[(x, y)] = agent
        self.world[x, y] = 1

        # Assign the first index where self.agent_indexes is not -1
        agent.index = self.population
        self.agent_indexes[agent.index] = agent.index

        self.agents.append(agent)
        self.population += 1

    def remove_agent(self, agent):
        
        if agent.get_position() not in self.position_agents:
            print("Agent not present at", agent.get_position())
            assert False

        self.agent_indexes[agent.index] =  -1
        self.population -= 1
        self.world[agent.get_position()] = 0
        self.agents.remove(agent)
        del self.position_agents[agent.get_position()]
        del agent

    def step(self, agent_actions):
        # Note: length of agent_actions should be equal to number of agents in the world
        
        # things that require resolution
        # 1. movement: a cell cannot have more than one agent, so we need to resolve this
        # 2. Kill: if agent 1 kills agent 2, agent 2 will not be able to act so we need to resolve this
            
        # we will create world copies to avoid changing the world while iterating
        new_world = self.world.copy()

        # resolve killing:
        for k, agent in enumerate(self.agents):

            if agent.get_responsiveness() < 0.5:
                continue
            
            if agent_actions[k]["Kill"] > 0:
                # get kill position
                x, y = agent.get_position()
                move_direction = self.direction_index[agent.get_direction()]
                kill_position = (x + move_direction[0], y + move_direction[1])

                # find agent at kill position
                if kill_position in self.position_agents and kill_position != (x, y):
                    killed_agent = self.position_agents[kill_position]
                    self.remove_agent(killed_agent)      
                    self.killings += 1

        # resolve movement, set pheromone and other actions
        for k, agent in enumerate(self.agents):
            
            responsivenes = agent.responsiveness
            x, y = agent.x, agent.y
            move_direction = self.direction_index[agent.get_direction()]
            random_direction = self.direction_index[np.random.randint(1, 9)]
            right_direction = np.array([move_direction[1], -move_direction[0]])

            if responsivenes >= 0.5:
                # compute move direction 
                move_fwd = agent_actions[k]["Mfd"] * move_direction
                move_rnd = agent_actions[k]["Mrn"] * random_direction
                move_rv = agent_actions[k]["Mrv"] * -move_direction
                move_rl = agent_actions[k]["MRL"] * right_direction
                move_x = agent_actions[k]["MX"] * self.direction_index[self.movement_names["east"]]
                move_y = agent_actions[k]["MY"] * self.direction_index[self.movement_names["north"]]

                # # resolve movement
                move = move_fwd + move_rnd + move_rv + move_rl + move_x + move_y 
                norm = np.linalg.norm(move)

                move_unit = np.round(move / (np.linalg.norm(move) + 10e-12)) # here x and y will be 0, 1 or -1

                # find new move direction
                new_x = int(x + move_unit[0])
                new_y = int(y + move_unit[1])
                new_x = (new_x if new_x >= 0 else 0) if new_x < self.world_size[0] else self.world_size[0] - 1
                new_y = (new_y if new_y >= 0 else 0) if new_y < self.world_size[1] else self.world_size[1] - 1

                if new_x == x and new_y == y:
                    # stay in same position
                    self.new_world[x, y] = 1
                else:
                    # check if new position is valid
                    world_val = self.world[new_x, new_y]
                    new_world_val = new_world[new_x, new_y]
                    block_val = self.blockage[new_x, new_y]
                    if world_val == 0 and new_world_val == 0 and block_val == 0:
                        # move agent
                        agent.set_position(new_x, new_y)
                        self.position_agents[(new_x, new_y)] = agent
                        del self.position_agents[(x, y)]
                        new_world[x, y] = 0
                        new_world[new_x, new_y] = 1
                        

            # update pheromone
            self.pheromone[x, y] += agent_actions[k]["SG"] * responsivenes
            
            # update agent age, oscillator, long probe distance and responsiveness
            agent.set_age(agent.age + 1)
            agent.set_oscillator(agent.oscillator + agent_actions[k]["OSC"])
            agent.set_long_probe_distance(agent.long_probe_distance + agent_actions[k]["LPD"])
            agent.set_responsiveness(agent.responsiveness + agent_actions[k]["Res"])


        # decay pheromone
        self.pheromone = self.pheromone * self.pheromone_decay_rate

        # # update world
        self.world = new_world
        self.population = len(self.agents)
        self.current_step += 1

        done = self.current_step >= self.max_steps_per_generation

        if done:
            self.destroy_unsafe_agents()

        return self.get_state(), 0, done, {}

    def render(self):

        # draw world, agents, danger_zone, blockage in single matrix and plot
        # world should be white, danger should be red, blockage should be gray, agents should be green

        # draw world
        world = np.zeros((self.world_size[0], self.world_size[1], 3)) + 255
        world[self.danger == 1] = [128, 20, 20]
        world[self.blockage == 1] = [80, 80, 80]
        world[self.world == 1] = [0, 128, 0]

        world = world.astype(np.uint8)

        return world

    def load_agents(self, dnas, generations):
        self.dna_bank = dnas
        self.generation_count = generations