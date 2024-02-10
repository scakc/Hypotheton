# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

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
        self.gene_length = gene_length

        # 9 movement directions
        self.movement_names = {
            "stay": 0, "west": 1, "north-west": 2, "north": 3, "north-east": 4, "east": 5, "south-east": 6, "south": 7, "south-west": 8
        }
        self.direction_index = {
            # each is corresponding movement in x and y as [x, y]
            0: [0, 0], 1: [-1, 0], 2: [-1, 1], 3: [0, 1], 4: [1, 1], 5: [1, 0], 6: [1, -1], 7: [0, -1], 8: [-1, -1]
        }

        self.pheromone_decay_rate = 0.9
        self.reset()
    
    def reset(self):
        
        self.step = 0
        self.world = np.zeros(self.world_size)
        self.pheromone = np.zeros(self.world_size)
        self.danger = np.zeros(self.world_size)
        self.blockage = np.zeros(self.world_size)
        
        # Add danger zone and blockage
        if self.env_type == "simple":
            self.danger[0:10, 0:10] = 1
            self.danger[0:10, -10:] = 1
            self.danger[-10:, 0:10] = 1
            self.danger[-10:, -10:] = 1
            self.blockage[0:10, :] = 1
            self.blockage[-10:, :] = 1
            self.blockage[:, 0:10] = 1
            self.blockage[:, -10:] = 1

        self.agents = []
        # create agents with mutated dna
        for i in range(self.max_population):
            random_dna = [format(np.random.randint(0, 16**8, dtype=np.int64) , '08x') for _ in range(self.gene_length)]
            self.add_agent(Agent(self.number_hidden_neurons, random_dna))

        # random placement of agents where there is no blockage
        self.population = 0
        self.position_agents = {}
        while self.population < self.max_population:
            x = np.random.randint(self.world_size[0])
            y = np.random.randint(self.world_size[1])
            if self.blockage[x, y] == 0 and self.world[x, y] == 0:
                self.world[x, y] = 1
                self.agents[self.population].set_position(x, y)
                self.position_agents[(x, y)] = self.agents[self.population]
                self.population += 1

        # def get_gradient(self, property):
        #     # get gradient of property
        #     gradient = np.gradient(property)
        #     return np.sqrt(gradient[0]**2 + gradient[1]**2)

    def get_state(self):
        # computes agent sensory input for each agent
        # self.pheromone_gradient = 
        agent_states = []
        for agent in self.agents:
            agent_features = self.get_features(agent)
            agent_states.append(agent_features)

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

    def set_box_value(self, property, x, y, value, box_size = [1,1,1,1]):
        # set sum of box_size x box_size cells around x, y

        x_start = x - box_size[0] if x - box_size[0] >= 0 else 0
        x_end = x + box_size[1] if x + box_size[1] < self.world_size[0] else self.world_size[0]
        y_start = y - box_size[2] if y - box_size[2] >= 0 else 0
        y_end = y + box_size[3] if y + box_size[3] < self.world_size[1] else self.world_size[1]

        property[x_start:x_end, y_start:y_end] = value

    def get_probe_value(self, property, x, y, direction, probe_distance = 5):

        probe_values = []
        for i in range(1, probe_distance + 1):
            probe_values.append(self.get_cell_value(property, x + i * direction[0], y + i * direction[1]))

        return probe_values

    def get_features(self, agent):
        
        x, y = agent.get_position()
        move_direction = self.direction_index[agent.get_direction()]

        env_features = {}
        left_move = [-move_direction[1], move_direction[0]]
        right_move = [move_direction[1], -move_direction[0]]
        forward_move = move_direction

        # pheromone gradient left - right
        left_peromone = self.get_cell_value(self.pheromone, x + left_move[0], y + left_move[1])
        right_peromone = self.get_cell_value(self.pheromone, x + right_move[0], y + right_move[1])
        env_features["Slr"] = left_peromone - right_peromone

        # pheromone gradient forward
        forward_peromone = self.get_cell_value(self.pheromone, x + forward_move[0], y + forward_move[1])
        current_peromone = self.get_cell_value(self.pheromone, x, y)
        env_features["Sfd"] = current_peromone - forward_peromone

        # pheromone density sum of surrounding 8 cells
        env_features["Sg"] = self.get_box_value(self.pheromone, x, y).mean()

        # age
        env_features["Age"] = agent.get_age()

        # random input
        env_features["Rnd"] = np.random.rand()

        # blockage left - right
        left_blockage = self.get_cell_value(self.blockage, x + left_move[0], y + left_move[1])
        right_blockage = self.get_cell_value(self.blockage, x + right_move[0], y + right_move[1])
        env_features["Blr"] = left_blockage - right_blockage # does not work if both are blocked (assuming will never happen)
        
        # oscillator
        env_features["Osc"] = agent.get_oscillator()

        # blockage forward
        env_features["Bfd"] = self.get_cell_value(self.blockage, x + forward_move[0], y + forward_move[1])

        # population gradient left - right
        left_population = self.get_cell_value(self.world, x + left_move[0], y + left_move[1])
        right_population = self.get_cell_value(self.world, x + right_move[0], y + right_move[1])
        env_features["Plr"] = left_population - right_population

        # population density sum of surrounding 8 cells
        env_features["Pop"] = self.get_box_value(self.world, x, y).mean()

        # population gradient forward
        forward_population = self.get_cell_value(self.world, x + forward_move[0], y + forward_move[1])
        current_population = self.get_cell_value(self.world, x, y)
        env_features["Pfd"] = forward_population - current_population

        # population long-range forward
        agent_long_range = np.round(agent.get_long_probe_distance())
        env_features["LPf"] = self.get_probe_value(self.world, x, y, forward_move, agent_long_range).sum()

        # last movement y
        current_position = agent.get_position()
        env_features["LMy"] = y - move_direction[1]

        # blockage long-range forward
        env_features["LBf"] = self.get_probe_value(self.blockage, x, y, forward_move, agent_long_range).sum()

        # last movement x
        env_features["LMx"] = x - move_direction[0]

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

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def step(self, agent_actions):
        # Note: length of agent_actions should be equal to number of agents in the world
        
        self.step += 1

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
                if kill_position in self.position_agents:
                    killed_agent = self.position_agents[kill_position]
                    self.remove_agent(killed_agent)
                    del self.position_agents[kill_position]
        
        # resolve movement, set pheromone and other actions
        for k, agent in enumerate(self.agents):
            
            responsivenes = agent.get_responsiveness()
            x, y = agent.get_position()
            move_direction = np.array(self.direction_index[agent.get_direction()])

            if responsivenes >= 0.5:
                # compute move direction 
                move_fwd = agent_actions[k]["Mfd"] * move_direction
                move_rnd = agent_actions[k]["Mrn"] * self.direction_index[np.random.randint(1, 9)]
                move_rv = agent_actions[k]["Mrv"] * -move_direction
                right_move = np.array([move_direction[1], -move_direction[0]])
                move_rl = agent_actions[k]["MRL"] * right_move
                move_x = agent_actions[k]["MX"] * self.direction_index[self.movement_names["east"]]
                move_y = agent_actions[k]["MY"] * self.direction_index[self.movement_names["north"]]

                # resolve movement
                move = move_fwd + move_rnd + move_rv + move_rl + move_x + move_y
                move_unit = np.round(move / np.linalg.norm(move)) # here x and y will be 0, 1 or -1

                # find new move direction
                new_x = int(x + move_unit[0])
                new_y = int(y + move_unit[1])

                # check if new position is valid
                world_val = self.get_cell_value(new_world, new_x, new_y, -1)
                if world_val == 0:
                    # move agent
                    new_world[x, y] = 0
                    new_world[new_x, new_y] = 1
                    agent.set_position(new_x, new_y)
                    self.position_agents[(new_x, new_y)] = agent
                    del self.position_agents[(x, y)]

            # decay pheromone
            self.pheromone = self.pheromone * self.pheromone_decay_rate
            # update pheromone
            self.set_box_value(self.pheromone, x, y, agent_actions[k]["SG"] * responsivenes)

            # update agent age, oscillator, long probe distance and responsiveness
            agent.set_age(agent.get_age() + 1)
            agent.set_oscillator(agent.get_oscillator() + agent_actions[k]["OSC"])
            agent.set_long_probe_distance(agent.get_long_probe_distance() + agent_actions[k]["LPD"])
            agent.set_responsiveness(agent.get_responsiveness() + agent_actions[k]["Res"])

        # update world
        self.world = new_world

        return self.get_state(), 0, self.step >= self.max_steps_per_generation, {}

    def render(self):
        # set figure size to be 10x10
        plt.figure(figsize=(10, 10))

        # draw world, agents, danger_zone, blockage in single matrix and plot
        world = self.world + self.danger + self.blockage
        plt.imshow(world, cmap='viridis')
        plt.show()

