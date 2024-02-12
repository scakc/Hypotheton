import numpy as np
import matplotlib.pyplot as plt 
import importlib
import concurrent.futures

import time

# import animation module here
import matplotlib.animation as animation
# from IPython.display import HTML, clear_output as clr
import pickle

import env as Environments
import agent as Agents
import sys
# importlib.reload(Environments)
# importlib.reload(Agents)

def save_generations(env, image_list, suffix = ''):
    agents = [agent.dna for agent in env.agents]
    generations = env.generation_count
    images = image_list

    pickle.dump(agents, open("saves/agents"+suffix+".pkl", "wb"))
    pickle.dump(generations, open("saves/generations"+suffix+".pkl", "wb"))
    pickle.dump(images, open("saves/images"+suffix+".pkl", "wb"))

def load_generations(suffix = ''):
    agents = pickle.load(open("saves/agents"+suffix+".pkl", "rb"))
    generations = pickle.load(open("saves/generations"+suffix+".pkl", "rb"))
    images = pickle.load(open("saves/images"+suffix+".pkl", "rb"))
    return agents, generations, images

if __name__ == '__main__':

    # get arguments

    # sample run evolution environment
    env = Environments.SimpleEvolutionEnv(max_steps_per_generation=200, population=1000, number_hidden_neurons = 4, gene_length = 16)
    
    # Initiate the evolution
    image_list = []
    state_time_lapses = []
    action_time_lapses = []
    t0 = time.time()
    output_keys = env.agents[0].output_index.keys()

    for gen in range(2):
        print_txt = ""
        print_txt += f"generation {gen}, "
        print_txt += "average time per step: states {}, actions {}, ".format(np.mean(state_time_lapses[-5:]), np.mean(action_time_lapses[-5:]))
        print_txt += f"survival_rate {env.survival_rate}, "
        # print_txt += '\r'
        
        print(print_txt)
        
        generation_images = []
        state_time_lapses = []
        action_time_lapses = []
        
        # lets time each step
        done = False
        max_step = 201
        steps = 0
        states = env.reset(keep_old_agents = True)
        while not done and steps < max_step:
            
            # print(print_txt + f"Steps:, {steps}")
            img = env.render()
            start = time.time()
            sample_actions = [env.agents[k].get_outputs(states[k]) for k in range(env.population)]
            # sample_actions = [{key: np.random.rand() for key in output_keys} for k in range(env.population)]
            action_time_lapses.append(time.time() - start)
            start = time.time()
            states, _, done, _ = env.step(sample_actions)
            state_time_lapses.append(time.time() - start)
            start = time.time()
            generation_images.append(img)
            steps += 1

        image_list.append(generation_images)
        
    print(f"Saving Generation, total_time {time.time() - t0}")
    save_generations(env, image_list, suffix = '_a')